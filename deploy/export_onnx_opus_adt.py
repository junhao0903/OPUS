"""Export an OPUS model's fixed-shape raw predictions to ONNX.

Voxel merging and thresholding are intentionally left outside the graph. The
exported graph accepts images plus calibration matrices and returns the final
decoder layer's point, class, and padding predictions.
"""

import argparse
import importlib
import sys
import types
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch import nn
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models.builder import build_detector
from models.csrc import wrapper as opus_sampling_wrapper
from models.utils import DUMP

MSMV_CUDA = opus_sampling_wrapper.MSMV_CUDA


def _export_decoder_forward(decoder, query_points, query_feat, mlvl_feats, img_metas):
    """Decoder forward that keeps calibration tensors in the ONNX graph."""
    cls_scores, refine_pts = [], []
    ego2img = torch.stack([meta['ego2img'] for meta in img_metas]).to(query_feat.dtype)
    occ2ego = torch.stack([meta['occ2ego'] for meta in img_metas]).to(query_feat.dtype)
    occ2img = ego2img @ occ2ego[:, None]

    # Keep this feature layout step in sync with OPUSTransformerDecoder.forward.
    for level, feat in enumerate(mlvl_feats):
        batch, num_temporal_views, grouped_channels, height, width = feat.shape
        views, frames, groups = decoder.num_views, decoder.num_frames, decoder.num_groups
        channels = grouped_channels // groups
        assert frames * views == num_temporal_views
        feat = feat.reshape(batch, frames, views, groups, channels, height, width)
        if MSMV_CUDA:
            feat = feat.permute(0, 1, 3, 2, 5, 6, 4)
            feat = feat.reshape(batch * frames * groups, views, height, width, channels)
        else:
            feat = feat.permute(0, 1, 3, 4, 2, 5, 6)
            feat = feat.reshape(batch * frames * groups, channels, views, height, width)
        mlvl_feats[level] = feat.contiguous()

    for layer_index, decoder_layer in enumerate(decoder.decoder_layers):
        DUMP.stage_count = layer_index
        query_points = query_points.detach()
        query_feat, cls_score, query_points = decoder_layer(
            query_points, query_feat, mlvl_feats, occ2img, img_metas)
        cls_scores.append(cls_score)
        refine_pts.append(query_points)
    return cls_scores, refine_pts


def _export_nan_to_num(x):
    """Replace non-finite values without exporting IsNaN or IsInf nodes.

    The MDC 610 BS9SX1AA operator package has no registered kernel for those
    two nodes. ``x == x`` is false only for NaN; after replacing NaNs with zero,
    Maximum/Minimum clamp infinities to the finite limits used by
    ``torch.nan_to_num``.
    """
    limits = torch.finfo(x.dtype)
    x = torch.where(x == x, x, x.new_tensor(0.0))
    x = torch.maximum(x, x.new_tensor(limits.min))
    return torch.minimum(x, x.new_tensor(limits.max))


def _export_transformer_forward(transformer, query_points, query_feat,
                                mlvl_feats, img_metas):
    """Keep transformer behavior while using ONNX-friendly finite checks."""
    cls_scores, refine_pts = transformer.decoder(
        query_points, query_feat, mlvl_feats, img_metas)
    cls_scores = [None if score is None else _export_nan_to_num(score)
                  for score in cls_scores]
    refine_pts = [None if points is None else _export_nan_to_num(points)
                  for points in refine_pts]
    return cls_scores, refine_pts


def _remove_sampling_noop_expands(onnx_model):
    """Remove fixed sampling Expand nodes that only preserve tensor shape.

    PyTorch exports these Expand nodes from broadcasting scalar divisors over
    gathered sampling coordinates. The target shape comes from the gathered
    coordinate tensor and has the same dimensions as the Div result. ATC's
    shape inference fails on this redundant dynamic Expand, so feed its input
    directly to consumers. Restrict the rewrite to the four known nodes in
    each decoder layer's sampling subgraph; other Expand nodes are untouched.
    """
    noop_names = {'Expand_20', 'Expand_27', 'Expand_34', 'Expand_41'}
    replacements = {}
    kept_nodes = []
    removed = 0

    for node in onnx_model.graph.node:
        is_sampling_noop = (
            node.op_type == 'Expand'
            and '/sampling/Expand_' in node.name
            and node.name.rsplit('/', 1)[-1] in noop_names
        )
        if not is_sampling_noop:
            kept_nodes.append(node)
            continue

        if len(node.input) != 2 or len(node.output) != 1:
            raise RuntimeError(f'Unexpected ONNX node structure: {node.name}')
        replacements[node.output[0]] = node.input[0]
        removed += 1

    if not removed:
        return onnx_model

    for node in kept_nodes:
        for index, value in enumerate(node.input):
            while value in replacements:
                value = replacements[value]
            node.input[index] = value
    for output in onnx_model.graph.output:
        value = output.name
        while value in replacements:
            value = replacements[value]
        output.name = value

    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(kept_nodes)
    print(f'Removed {removed} redundant sampling Expand nodes')
    return onnx_model


def _staticize_sampling_shapes(onnx_model):
    """Fold fixed sampling Expand shapes and Range sequences for ATC.

    The exported graph has fixed input dimensions. Evaluate only the small
    shape/index tensors with ONNX Runtime, then replace their dynamic Shape /
    Where / Range paths by constants. This avoids ATC shape inference failures
    on those paths while preserving the values produced for the fixed graph.
    """
    import copy
    import numpy as np
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    probe_names = []
    for node in onnx_model.graph.node:
        if '/sampling/' not in node.name:
            continue
        names = []
        if node.op_type == 'Expand':
            names.append(node.input[1])
        elif node.op_type == 'Range':
            names.append(node.output[0])
        for name in names:
            if name not in probe_names:
                probe_names.append(name)
    if not probe_names:
        return onnx_model

    original_outputs = [copy.deepcopy(output)
                        for output in onnx_model.graph.output]
    existing_outputs = {output.name for output in onnx_model.graph.output}
    for name in probe_names:
        if name not in existing_outputs:
            onnx_model.graph.output.append(
                helper.make_tensor_value_info(name, TensorProto.INT64, [None]))

    session = ort.InferenceSession(
        onnx_model.SerializeToString(), providers=['CPUExecutionProvider'])
    feeds = {}
    dtype_map = {
        'tensor(float)': np.float32,
        'tensor(double)': np.float64,
        'tensor(int64)': np.int64,
        'tensor(int32)': np.int32,
        'tensor(bool)': np.bool_,
    }
    for model_input in session.get_inputs():
        shape = [dim if isinstance(dim, int) and dim > 0 else 1
                 for dim in model_input.shape]
        dtype = dtype_map.get(model_input.type)
        if dtype is None:
            raise RuntimeError(
                f'Cannot make a probe input for {model_input.name} '
                f'with type {model_input.type}')
        feeds[model_input.name] = np.zeros(shape, dtype=dtype)
    values = dict(zip(probe_names, session.run(probe_names, feeds)))

    onnx_model.graph.ClearField('output')
    onnx_model.graph.output.extend(original_outputs)

    initializer_names = {initializer.name
                         for initializer in onnx_model.graph.initializer}

    def add_constant(name, value):
        if name in initializer_names:
            raise RuntimeError(f'Duplicate ONNX initializer name: {name}')
        onnx_model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(value), name=name))
        initializer_names.add(name)

    kept_nodes = []
    expanded = 0
    folded_ranges = 0
    for node in onnx_model.graph.node:
        if '/sampling/' in node.name and node.op_type == 'Expand':
            constant_name = node.name.replace('/', '_') + '_static_shape'
            add_constant(constant_name, values[node.input[1]])
            node.input[1] = constant_name
            expanded += 1
        if '/sampling/' in node.name and node.op_type == 'Range':
            add_constant(node.output[0], values[node.output[0]])
            folded_ranges += 1
            continue
        kept_nodes.append(node)

    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(kept_nodes)
    print('Staticized sampling shapes:', expanded,
          'Expand targets and', folded_ranges, 'Range nodes')
    return onnx_model


def _flatten_sampling_matmuls(onnx_model, batch, frames, views, queries,
                              points):
    """Flatten seven-dimensional projection MatMuls to ATC's supported rank."""
    from onnx import helper, numpy_helper
    import numpy as np

    batch_product = batch * frames * views * queries * points
    full_shape = [batch, frames, views, queries, points]
    constants = {
        'a': [batch_product, 4, 4],
        'b': [batch_product, 4, 1],
        'out': full_shape + [4, 1],
    }
    initializer_names = {initializer.name
                         for initializer in onnx_model.graph.initializer}

    def add_constant(name, value):
        if name in initializer_names:
            raise RuntimeError(f'Duplicate ONNX initializer name: {name}')
        onnx_model.graph.initializer.append(numpy_helper.from_array(
            np.asarray(value, dtype=np.int64), name=name))
        initializer_names.add(name)

    new_nodes = []
    rewritten = 0
    for node in onnx_model.graph.node:
        if node.op_type != 'MatMul' or '/sampling/MatMul' not in node.name:
            new_nodes.append(node)
            continue

        prefix = node.name.rsplit('/', 1)[0]
        original_output = node.output[0]
        suffix = original_output.replace('/', '_')
        names = {key: f'{suffix}_flat_shape_{key}' for key in constants}
        for key, value in constants.items():
            add_constant(names[key], value)

        flat_a = original_output + '_flat_a'
        flat_b = original_output + '_flat_b'
        flat_output = original_output + '_flat_output'
        new_nodes.append(helper.make_node(
            'Reshape', [node.input[0], names['a']], [flat_a],
            name=prefix + '/MatMul_flatten_a'))
        new_nodes.append(helper.make_node(
            'Reshape', [node.input[1], names['b']], [flat_b],
            name=prefix + '/MatMul_flatten_b'))
        node.input[0], node.input[1] = flat_a, flat_b
        node.output[0] = flat_output
        new_nodes.append(node)
        new_nodes.append(helper.make_node(
            'Reshape', [flat_output, names['out']], [original_output],
            name=prefix + '/MatMul_restore_shape'))
        rewritten += 1

    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(new_nodes)
    print(f'Flattened {rewritten} sampling MatMul nodes for ATC')
    return onnx_model


def _convert_grid_sample_to_mdc_gridsample3d(onnx_model):
    """Rewrite standard 5D GridSample nodes to the MDC GridSample3d layout.

    Standard GridSample uses feature ``[N,C,D,H,W]`` and grid
    ``[N,Dout,Hout,Wout,3]``. MDC GridSample3d expects feature
    ``[N,D,C1,H,W,16]``, grid ``[N,3,Dout,Hout,Wout]``, and output
    ``[N,Dout,C1,Hout,Wout,16]``. This pass inserts the required Reshape and
    Transpose nodes, and casts data to the MDC operator's float16 interface.
    Static dimensions and channels divisible by 16 are required.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    value_info = {}
    for value in (list(onnx_model.graph.input)
                  + list(onnx_model.graph.output)
                  + list(onnx_model.graph.value_info)):
        value_info[value.name] = value

    def get_shape(name):
        value = value_info.get(name)
        if value is None or not value.type.HasField('tensor_type'):
            return None
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField('shape'):
            return None
        dims = []
        for dim in tensor_type.shape.dim:
            if not dim.HasField('dim_value') or dim.dim_value <= 0:
                return None
            dims.append(int(dim.dim_value))
        return dims

    def get_dtype(name):
        value = value_info.get(name)
        if value is None or not value.type.HasField('tensor_type'):
            return TensorProto.FLOAT
        return value.type.tensor_type.elem_type

    initializer_names = {initializer.name
                         for initializer in onnx_model.graph.initializer}

    def add_shape_constant(name, shape):
        if name in initializer_names:
            raise RuntimeError(f'Duplicate ONNX initializer name: {name}')
        onnx_model.graph.initializer.append(numpy_helper.from_array(
            np.asarray(shape, dtype=np.int64), name=name))
        initializer_names.add(name)

    new_nodes = []
    converted = 0
    for node in onnx_model.graph.node:
        if node.op_type != 'GridSample' or node.domain not in ('', 'ai.onnx'):
            new_nodes.append(node)
            continue
        if len(node.input) != 2 or len(node.output) != 1:
            raise RuntimeError(f'Unexpected GridSample node: {node.name}')

        feature_shape = get_shape(node.input[0])
        grid_shape = get_shape(node.input[1])
        if (feature_shape is None or grid_shape is None
                or len(feature_shape) != 5 or len(grid_shape) != 5):
            raise RuntimeError(
                f'{node.name}: MDC conversion requires static rank-5 feature '
                'and grid inputs')
        batch, channels, depth, height, width = feature_shape
        grid_batch, out_depth, out_height, out_width, coordinates = grid_shape
        if grid_batch != batch or coordinates != 3:
            raise RuntimeError(
                f'{node.name}: expected grid shape [N,Dout,Hout,Wout,3], '
                f'got {grid_shape}')
        if channels % 16:
            raise RuntimeError(
                f'{node.name}: feature channel count {channels} is not '
                'divisible by MDC C0=16')

        expected_output_shape = [
            batch, channels, out_depth, out_height, out_width]
        actual_output_shape = get_shape(node.output[0])
        if (actual_output_shape is not None
                and actual_output_shape != expected_output_shape):
            raise RuntimeError(
                f'{node.name}: expected standard GridSample output '
                f'{expected_output_shape}, got {actual_output_shape}')

        attrs = {attr.name: helper.get_attribute_value(attr)
                 for attr in node.attribute}
        mode = attrs.get('mode', b'bilinear')
        if isinstance(mode, bytes):
            mode = mode.decode('ascii')
        if mode == 'linear':
            mode = 'bilinear'
        if mode not in ('bilinear', 'nearest'):
            raise RuntimeError(
                f'{node.name}: MDC GridSample3d does not support mode={mode}')

        padding_mode = attrs.get('padding_mode', b'zeros')
        if isinstance(padding_mode, bytes):
            padding_mode = padding_mode.decode('ascii')
        if padding_mode not in ('zeros', 'border', 'reflection'):
            raise RuntimeError(
                f'{node.name}: unsupported padding_mode={padding_mode}')
        align_corners = bool(attrs.get('align_corners', 0))

        prefix = node.name or node.output[0]
        prefix = prefix.replace('/', '_').replace(':', '_')
        names = {
            'feature_shape': prefix + '_mdc_feature_shape',
            'feature_reshaped': prefix + '_mdc_feature_reshaped',
            'feature_layout': prefix + '_mdc_feature_layout',
            'grid_layout': prefix + '_mdc_grid_layout',
            'sampled': prefix + '_mdc_sampled',
            'output_layout': prefix + '_standard_output_layout',
            'output_shape': prefix + '_standard_output_shape',
        }
        c1 = channels // 16
        add_shape_constant(names['feature_shape'],
                           [batch, c1, 16, depth, height, width])
        add_shape_constant(names['output_shape'], expected_output_shape)

        feature_input = node.input[0]
        if get_dtype(feature_input) != TensorProto.FLOAT16:
            cast_feature = prefix + '_mdc_feature_fp16'
            new_nodes.append(helper.make_node(
                'Cast', [feature_input], [cast_feature],
                name=prefix + '_mdc_cast_feature_to_fp16',
                to=TensorProto.FLOAT16))
            feature_input = cast_feature

        feature_reshaped = names['feature_reshaped']
        feature_layout = names['feature_layout']
        new_nodes.append(helper.make_node(
            'Reshape', [feature_input, names['feature_shape']],
            [feature_reshaped], name=prefix + '_mdc_split_channels'))
        new_nodes.append(helper.make_node(
            'Transpose', [feature_reshaped], [feature_layout],
            name=prefix + '_mdc_feature_transpose',
            perm=[0, 3, 1, 4, 5, 2]))

        grid_input = node.input[1]
        if get_dtype(grid_input) != TensorProto.FLOAT16:
            cast_grid = prefix + '_mdc_grid_fp16'
            new_nodes.append(helper.make_node(
                'Cast', [grid_input], [cast_grid],
                name=prefix + '_mdc_cast_grid_to_fp16',
                to=TensorProto.FLOAT16))
            grid_input = cast_grid
        grid_layout = names['grid_layout']
        new_nodes.append(helper.make_node(
            'Transpose', [grid_input], [grid_layout],
            name=prefix + '_mdc_grid_transpose', perm=[0, 4, 1, 2, 3]))

        sampled = names['sampled']
        new_nodes.append(helper.make_node(
            'GridSample3d', [feature_layout, grid_layout], [sampled],
            name=prefix + '_MDC_GridSample3d',
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners))

        output_layout = names['output_layout']
        output_dtype = get_dtype(node.output[0])
        output_reshape = (node.output[0] if output_dtype == TensorProto.FLOAT16
                          else prefix + '_mdc_output_fp16')
        new_nodes.append(helper.make_node(
            'Transpose', [sampled], [output_layout],
            name=prefix + '_mdc_output_transpose', perm=[0, 2, 5, 1, 3, 4]))
        new_nodes.append(helper.make_node(
            'Reshape', [output_layout, names['output_shape']], [output_reshape],
            name=prefix + '_mdc_restore_standard_output'))
        if output_dtype != TensorProto.FLOAT16:
            new_nodes.append(helper.make_node(
                'Cast', [output_reshape], [node.output[0]],
                name=prefix + '_mdc_cast_output_from_fp16', to=output_dtype))
        converted += 1

    if not converted:
        print('No standard 5D GridSample nodes found for MDC conversion')
        return onnx_model

    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(new_nodes)
    print(f'Converted {converted} standard GridSample nodes to MDC GridSample3d')
    return onnx_model


def _convert_gather_elements_to_mdc(onnx_model):
    """Map standard ONNX GatherElements nodes to MDCGatherElements.

    The MDC custom operator accepts float16/float32 data and int32 indices.
    ONNX GatherElements commonly receives int64 indices from PyTorch export, so
    cast that input to int32 and preserve the original axis and output value.
    """
    import onnx
    from onnx import TensorProto, helper

    nodes = [node for node in onnx_model.graph.node
             if node.op_type == 'GatherElements'
             and node.domain in ('', 'ai.onnx')]
    if not nodes:
        print('No standard GatherElements nodes found for MDC conversion')
        return onnx_model

    inferred_model = onnx.shape_inference.infer_shapes(
        onnx_model, strict_mode=False)
    tensor_info = {}
    for value in (list(inferred_model.graph.input)
                  + list(inferred_model.graph.output)
                  + list(inferred_model.graph.value_info)):
        tensor_info[value.name] = value

    value_names = set(tensor_info)
    value_names.update(output for node in onnx_model.graph.node
                       for output in node.output)
    rewritten_nodes = []
    converted = 0
    for node in onnx_model.graph.node:
        if (node.op_type != 'GatherElements'
                or node.domain not in ('', 'ai.onnx')):
            rewritten_nodes.append(node)
            continue
        if len(node.input) != 2 or len(node.output) != 1:
            raise RuntimeError(f'Unexpected GatherElements node: {node.name}')

        data_info = tensor_info.get(node.input[0])
        if data_info is not None and data_info.type.HasField('tensor_type'):
            data_type = data_info.type.tensor_type.elem_type
            if data_type not in (TensorProto.FLOAT16, TensorProto.FLOAT):
                raise RuntimeError(
                    f'{node.name}: MDCGatherElements data must be float16 or '
                    f'float32, got TensorProto dtype {data_type}')

        prefix = (node.name or node.output[0]).replace('/', '_').replace(':', '_')
        cast_indices = prefix + '_mdc_indices_int32'
        if cast_indices in value_names:
            raise RuntimeError(f'Duplicate ONNX value name: {cast_indices}')
        rewritten_nodes.append(helper.make_node(
            'Cast', [node.input[1]], [cast_indices],
            name=prefix + '_cast_indices_to_int32', to=TensorProto.INT32))
        value_names.add(cast_indices)
        node.input[1] = cast_indices
        node.op_type = 'MDCGatherElements'
        rewritten_nodes.append(node)
        converted += 1

    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(rewritten_nodes)
    print(f'Converted {converted} GatherElements nodes to MDCGatherElements')
    return onnx_model


def _rewrite_sampling_argmax_indices(onnx_model, num_views):
    """Avoid unsupported int64 ArgMax/index arithmetic in sampling subgraphs.

    ONNX ArgMax produces int64 indices, while this MDC operator package only
    provides ArgMaxV2 with int32 output. Replace the first-valid-view ArgMax
    with an equivalent fixed-view boolean selection, then perform sampling
    Add/Mul index arithmetic in int32. Cast values back to int64 for consumers
    outside that index path so shape calculations retain their original dtype.
    """
    import copy
    from onnx import TensorProto, helper, shape_inference

    onnx_model = shape_inference.infer_shapes(onnx_model)
    value_types = {}
    for value in (list(onnx_model.graph.input)
                  + list(onnx_model.graph.output)
                  + list(onnx_model.graph.value_info)):
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type:
            value_types[value.name] = tensor_type.elem_type
    for initializer in onnx_model.graph.initializer:
        value_types[initializer.name] = initializer.data_type

    # Build replacement subgraphs for ArgMax. valid_mask contains only 0/1,
    # and torch.argmax selects the first maximum; for an all-zero mask it
    # returns view 0, which is preserved by the selected0 fallback.
    argmax_replacements = {}
    argmax_nodes = [node for node in onnx_model.graph.node
                    if node.op_type == 'ArgMax'
                    and '/sampling/' in node.name]
    for node in argmax_nodes:
        attrs = {attr.name: helper.get_attribute_value(attr)
                 for attr in node.attribute}
        if attrs.get('axis', -1) not in (-1, 4) or attrs.get('keepdims', 0):
            raise RuntimeError(
                f'Unexpected sampling ArgMax attributes: {node.name} {attrs}')
        prefix = node.name + '_mdc_first_valid'
        zero_name = prefix + '_zero'
        zero = helper.make_tensor(zero_name, TensorProto.FLOAT, [], [0.0])
        onnx_model.graph.initializer.append(zero)
        view_valid = []
        generated = []
        for view in range(num_views):
            index_name = prefix + f'_view_{view}'
            index = helper.make_tensor(
                index_name, TensorProto.INT32, [], [view])
            onnx_model.graph.initializer.append(index)
            value_name = prefix + f'_value_{view}'
            valid_name = prefix + f'_valid_{view}'
            generated.append(helper.make_node(
                'Gather', [node.input[0], index_name], [value_name],
                name=prefix + f'_gather_{view}', axis=-1))
            generated.append(helper.make_node(
                'Greater', [value_name, zero_name], [valid_name],
                name=prefix + f'_greater_{view}'))
            view_valid.append(valid_name)

        any_valid = view_valid[0]
        for view, valid_name in enumerate(view_valid[1:], start=1):
            combined = prefix + f'_any_valid_{view}'
            generated.append(helper.make_node(
                'Or', [any_valid, valid_name], [combined],
                name=prefix + f'_or_valid_{view}'))
            any_valid = combined

        no_valid = prefix + '_no_valid'
        generated.append(helper.make_node(
            'Not', [any_valid], [no_valid], name=prefix + '_not_valid'))
        selected = []
        selected0 = prefix + '_selected_0'
        generated.append(helper.make_node(
            'Or', [view_valid[0], no_valid], [selected0],
            name=prefix + '_select_0'))
        selected.append(selected0)
        seen_valid = view_valid[0]
        for view in range(1, num_views):
            not_seen = prefix + f'_not_seen_{view}'
            selected_view = prefix + f'_selected_{view}'
            generated.append(helper.make_node(
                'Not', [seen_valid], [not_seen],
                name=prefix + f'_not_seen_node_{view}'))
            generated.append(helper.make_node(
                'And', [view_valid[view], not_seen], [selected_view],
                name=prefix + f'_select_node_{view}'))
            selected.append(selected_view)
            if view < num_views - 1:
                next_seen = prefix + f'_seen_{view}'
                generated.append(helper.make_node(
                    'Or', [seen_valid, view_valid[view]], [next_seen],
                    name=prefix + f'_seen_node_{view}'))
                seen_valid = next_seen

        cast_names = []
        for view, selected_name in enumerate(selected):
            cast_name = prefix + f'_selected_i32_{view}'
            generated.append(helper.make_node(
                'Cast', [selected_name], [cast_name],
                name=prefix + f'_cast_selected_{view}',
                to=TensorProto.INT32))
            cast_names.append(cast_name)
        # The selected-view booleans are one-hot flags. Start from zero so
        # view 0 contributes index 0 (including the all-invalid fallback).
        result = prefix + '_index_zero_i32'
        onnx_model.graph.initializer.append(helper.make_tensor(
            result, TensorProto.INT32, [], [0]))
        for view in range(1, num_views):
            term = prefix + f'_term_{view}'
            index_name = prefix + f'_index_i32_{view}'
            index = helper.make_tensor(
                index_name, TensorProto.INT32, [], [view])
            onnx_model.graph.initializer.append(index)
            generated.append(helper.make_node(
                'Mul', [cast_names[view], index_name], [term],
                name=prefix + f'_mul_index_{view}'))
            next_result = (prefix + '_result_i32' if view == num_views - 1
                           else prefix + f'_sum_{view}')
            generated.append(helper.make_node(
                'Add', [result, term], [next_result],
                name=prefix + f'_sum_index_{view}'))
            result = next_result
        generated.append(helper.make_node(
            'Cast', [result], [node.output[0]],
            name=prefix + '_cast_result_int64', to=TensorProto.INT64))
        argmax_replacements[node.name] = generated

    if argmax_replacements:
        rewritten = []
        for node in onnx_model.graph.node:
            replacement = argmax_replacements.get(node.name)
            rewritten.extend(replacement if replacement is not None else [node])
        del onnx_model.graph.node[:]
        onnx_model.graph.node.extend(rewritten)
        print(f'Replaced {len(argmax_nodes)} sampling ArgMax nodes with '
              'first-valid-view selection')

    # Re-infer types after replacing ArgMax before identifying index math.
    onnx_model = shape_inference.infer_shapes(onnx_model)
    value_types = {}
    for value in (list(onnx_model.graph.input)
                  + list(onnx_model.graph.output)
                  + list(onnx_model.graph.value_info)):
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type:
            value_types[value.name] = tensor_type.elem_type
    for initializer in onnx_model.graph.initializer:
        value_types[initializer.name] = initializer.data_type

    index_math = [node for node in onnx_model.graph.node
                  if '/sampling/' in node.name
                  and node.op_type in ('Add', 'Mul')
                  and len(node.output) == 1
                  and value_types.get(node.output[0]) == TensorProto.INT64]
    if not index_math:
        return onnx_model

    math_outputs = {node.output[0]: node.output[0] + '_mdc_int32'
                    for node in index_math}
    math_names = {node.name for node in index_math}
    consumers = {}
    for node in onnx_model.graph.node:
        for input_index, value in enumerate(node.input):
            consumers.setdefault(value, []).append((node, input_index))
    graph_outputs = {value.name for value in onnx_model.graph.output}
    casts = {}
    rewritten = []
    changed = 0

    def cast_value(value, target_type, target_name):
        key = (value, target_type)
        if key not in casts:
            casts[key] = target_name
            rewritten.append(helper.make_node(
                'Cast', [value], [target_name],
                name=target_name + '_cast', to=target_type))
        return casts[key]

    for node in onnx_model.graph.node:
        if node.name not in math_names:
            # Use int32 results for downstream integer arithmetic and Gather
            # indices; preserve int64 for all other existing consumers.
            for input_index, value in enumerate(node.input):
                if value not in math_outputs:
                    continue
                if (node.name in math_names
                        or (node.op_type in ('Gather', 'GatherElements', 'GatherND')
                            and input_index == 1)):
                    node.input[input_index] = math_outputs[value]
            rewritten.append(node)
            continue

        new_node = copy.deepcopy(node)
        new_inputs = []
        for input_index, value in enumerate(node.input):
            if value in math_outputs:
                new_inputs.append(math_outputs[value])
            elif value_types.get(value) == TensorProto.INT64:
                cast_name = value + '_sampling_i32'
                new_inputs.append(cast_value(
                    value, TensorProto.INT32, cast_name))
            else:
                new_inputs.append(value)
        new_node.input[:] = new_inputs
        new_node.output[0] = math_outputs[node.output[0]]
        new_node.name = node.name + '_mdc_int32'
        rewritten.append(new_node)

        needs_int64 = (node.output[0] in graph_outputs or any(
            consumer.name not in math_names
            and not (consumer.op_type in ('Gather', 'GatherElements', 'GatherND')
                     and input_index == 1)
            for consumer, input_index in consumers.get(node.output[0], [])))
        if needs_int64:
            rewritten.append(helper.make_node(
                'Cast', [math_outputs[node.output[0]]], [node.output[0]],
                name=node.name + '_mdc_cast_back_int64',
                to=TensorProto.INT64))
        changed += 1

    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(rewritten)
    onnx_model = shape_inference.infer_shapes(onnx_model)
    print(f'Rewrote {changed} sampling int64 Add/Mul nodes to int32')
    return onnx_model


def msmv_sampling_pytorch_cpu(mlvl_feats, sampling_locations, scale_weights):
    """用相机视角索引和四邻点 gather 实现 CPU ONNX 兼容的 MSMV 采样。

    原始算子在 [B, C, N, H, W] 特征上用 (x, y, view) 执行 5D
    grid_sample；目标 ONNX Runtime CPU 不支持该 5D 节点。OPUS 的 view 坐标
    是离散相机索引 k/(N-1)，所以可先选定相机，再按 x/y 双线性插值所需的
    四个邻点 gather 特征值。这样既避免 5D GridSample，也避免对所有相机
    重复采样。x/y 是连续归一化坐标；view 必须是离散相机索引，不支持视角
    维连续插值。关闭 MSMV CUDA 算子时使用此实现。
    """
    B, C, N, _, _ = mlvl_feats[0].shape
    _, Q, P, _ = sampling_locations.shape

    # view 是 [0, 1] 范围内的离散相机编号，恢复成 0 到 N-1 的整数索引。
    view_index = torch.round(
        sampling_locations[..., 2] * (N - 1)).to(torch.int32)
    final = torch.zeros([B, C, Q, P], device=mlvl_feats[0].device)

    for lvl, feat in enumerate(mlvl_feats):
        _, _, _, H, W = feat.shape

        # 将 [0, 1] 的图像坐标换算成当前特征层的像素坐标。
        # floor 得到左上邻点，x1/y1 是相邻的右侧/下侧像素。
        x = sampling_locations[..., 0] * (W - 1)
        y = sampling_locations[..., 1] * (H - 1)
        x0, y0 = torch.floor(x), torch.floor(y)
        x1, y1 = x0 + 1, y0 + 1
        wx, wy = x - x0, y - y0

        # 把相机、高、宽展平，后续用 view 和像素坐标构造一维 gather 索引。
        feat = feat.reshape(B, C, N * H * W)

        # 按双线性插值顺序收集左上、右上、左下、右下四个邻点。
        x_indices = torch.stack((x0, x1, x0, x1), dim=-1)
        y_indices = torch.stack((y0, y0, y1, y1), dim=-1)

        # grid_sample 的零填充语义要求越界邻点贡献为 0。索引本身需要
        # clamp 到有效范围以便 gather，因此另存 valid 掩码来清除越界值。
        valid = ((x_indices >= 0) & (x_indices < W)
                 & (y_indices >= 0) & (y_indices < H))
        indices = (view_index[..., None] * (H * W)
                   + y_indices.clamp(0, H - 1).to(torch.int32) * W
                   + x_indices.clamp(0, W - 1).to(torch.int32))

        # 索引计算保持 int32；只在 Gather 的 index_select 边界转为 int64。
        # index_select 导出为 ONNX Gather，避免 torch.gather 生成不支持的
        # GatherElements，也不需要把索引扩展到每个通道。
        indices = indices.reshape(B, Q * P * 4).to(torch.int64)
        values = torch.stack([
            torch.index_select(feat[batch_index], 1, indices[batch_index])
            for batch_index in range(B)
        ], dim=0).reshape(B, C, Q, P, 4)

        # 根据采样点在像素格中的小数偏移，计算四个邻点各自的双线性权重。
        weights = torch.stack((
            (1 - wx) * (1 - wy),
            wx * (1 - wy),
            (1 - wx) * wy,
            wx * wy,
        ), dim=-1)

        # 屏蔽越界邻点，乘权重并求和，得到当前特征层的采样值。
        level_sample = (
            values * valid[:, None].to(values.dtype)
            * weights[:, None]
        ).sum(dim=-1)

        # 对当前层采样结果乘以模型预测的层级权重，再累加各层结果。
        level_sample = level_sample * scale_weights[..., lvl].reshape(B, 1, Q, P)
        final = final + level_sample

    # 与原 MSMV 采样接口一致，返回 [B, Q, C, P]。
    return final.permute(0, 2, 1, 3)


def configure_msmv_cuda(device, requested):
    """Configure custom MSMV dispatch and return whether CUDA will be used."""
    global MSMV_CUDA
    extension_available = opus_sampling_wrapper.MSMV_CUDA
    if requested == 'auto':
        use_msmv_cuda = device.type == 'cuda' and extension_available
    else:
        use_msmv_cuda = requested == 'true'

    if use_msmv_cuda and device.type != 'cuda':
        raise ValueError('--use-msmv-cuda true requires --device cuda')
    if use_msmv_cuda and not extension_available:
        raise RuntimeError('MSMV CUDA was requested, but the CUDA extension is unavailable')

    MSMV_CUDA = use_msmv_cuda
    opus_sampling_wrapper.MSMV_CUDA = use_msmv_cuda
    if not use_msmv_cuda:
        # Use standard 2D GridSample nodes, including CUDA tracing for CPU targets.
        opus_sampling_wrapper.msmv_sampling_pytorch = msmv_sampling_pytorch_cpu
    return use_msmv_cuda


class OPUSOnnxExportWrapper(nn.Module):
    """Expose a lightweight single-frame OPUS inference benchmark graph."""

    def __init__(self, model, image_height, image_width):
        super().__init__()
        self.model = model
        self.image_height = image_height
        self.image_width = image_width
        self.num_frames = int(model.pts_bbox_head.transformer.decoder.num_frames)
        self.num_views = int(model.pts_bbox_head.transformer.decoder.num_views)
        # The regular decoder converts metadata through NumPy. Replace only
        # this instance's forward while exporting so calibration stays tensor IO.
        decoder = self.model.pts_bbox_head.transformer.decoder
        decoder.forward = types.MethodType(_export_decoder_forward, decoder)
        transformer = self.model.pts_bbox_head.transformer
        transformer.forward = types.MethodType(
            _export_transformer_forward, transformer)

    def forward(self, img, ego2img, occ2ego):
        # This benchmark-only graph uses the current six camera images. It
        # duplicates their extracted features to fill the configured temporal
        # window; it does not model historical feature fusion or cache state.
        batch = img.shape[0]
        assert img.shape[1] == self.num_views

        img_metas = []
        for index in range(batch):
            img_metas.append({
                'ego2img': ego2img[index].repeat(self.num_frames, 1, 1),
                'occ2ego': occ2ego[index],
                'img_shape': [(self.image_height, self.image_width, 3)] * (self.num_frames * self.num_views),
                'ori_shape': [(self.image_height, self.image_width, 3)] * (self.num_frames * self.num_views),
                'pad_shape': [(self.image_height, self.image_width, 3)] * (self.num_frames * self.num_views),
            })

        current_feats = self.model.extract_feat(img, img_metas)
        img_feats = [
            feat.unsqueeze(1)
                .expand(-1, self.num_frames, -1, -1, -1, -1)
                .reshape(batch, self.num_frames * self.num_views, *feat.shape[2:])
            for feat in current_feats
        ]

        predictions = self.model.pts_bbox_head(img_feats, img_metas)
        # The evaluation-mode transformer only emits its feature prediction
        # from the final decoder layer. Export raw tensors and post-process
        # voxel aggregation outside ONNX (torch.unique has dynamic output size).
        return (
            predictions['all_refine_pts'][-1],
            predictions['all_cls_scores'][-1],
            predictions['all_pad_scores'][-1],
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Export OPUS to ONNX')
    parser.add_argument('--config', '--cfg', dest='config', required=True,
                        help='MMCV config file for the OPUS model')
    parser.add_argument('--weights', '--ckpt', dest='weights', required=True,
                        help='Model checkpoint')
    parser.add_argument('--output', '--save_onnx', dest='output', required=True,
                        help='Output ONNX path')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu',
                        help='Device used for tracing; defaults to CPU to reduce GPU OOM risk')
    parser.add_argument('--use-msmv-cuda', choices=('auto', 'true', 'false'),
                        default='auto',
                        help=('Select the custom MSMV CUDA sampling op. auto disables it '
                              'on CPU and follows extension availability on CUDA.'))
    parser.add_argument('--opset', type=int, default=16)
    parser.add_argument('--simplify', action='store_true',
                        help='Run onnxsim after export (uses additional memory)')
    parser.add_argument('--batch-size', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.weights).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Register repository model and dataset modules before building from config.
    importlib.import_module('models')
    importlib.import_module('loaders')
    cfg = Config.fromfile(str(config_path))

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda was requested but CUDA is unavailable')
    device = torch.device(args.device)
    use_msmv_cuda = configure_msmv_cuda(device, args.use_msmv_cuda)
    print('MSMV CUDA sampling:', use_msmv_cuda)

    model = build_detector(cfg.model)
    load_checkpoint(model, str(checkpoint_path), map_location='cpu', strict=False)
    model.to(device).eval()
    model.fp16_enabled = False
    num_frames = int(cfg.model.pts_bbox_head.transformer.num_frames)
    num_views = int(getattr(cfg.model.pts_bbox_head.transformer, 'num_views', 6))
    num_groups = int(getattr(cfg.model.pts_bbox_head.transformer, 'num_groups', 4))
    num_points = int(getattr(cfg.model.pts_bbox_head.transformer, 'num_points', 4))
    num_queries = int(cfg.model.pts_bbox_head.num_query)
    wrapper = OPUSOnnxExportWrapper(
        model,
        image_height=int(cfg.ida_aug_conf['final_dim'][0]),
        image_width=int(cfg.ida_aug_conf['final_dim'][1]),
    ).to(device).eval()

    batch = args.batch_size
    height = int(cfg.ida_aug_conf['final_dim'][0])
    width = int(cfg.ida_aug_conf['final_dim'][1])
    dummy_img = torch.zeros(batch, num_views, 3, height, width, device=device)
    dummy_ego2img = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(
        batch, num_views, 1, 1)
    dummy_occ2ego = torch.eye(4, device=device).view(1, 4, 4).repeat(batch, 1, 1)
    export_inputs = (dummy_img, dummy_ego2img, dummy_occ2ego)

    print('Export device:', device)
    print('Input image shape:', tuple(dummy_img.shape))
    print('Writing ONNX:', output_path)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            export_inputs,
            str(output_path),
            input_names=['img', 'ego2img', 'occ2ego'],
            output_names=['refine_points', 'class_scores', 'pad_scores'],
            opset_version=args.opset,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            keep_initializers_as_inputs=False,
        )

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    if args.simplify:
        from onnxsim import simplify
        onnx_model, ok = simplify(onnx_model)
        if not ok:
            raise RuntimeError('onnxsim could not validate the simplified graph')
    onnx_model = _convert_grid_sample_to_mdc_gridsample3d(onnx_model)
    onnx_model = _remove_sampling_noop_expands(onnx_model)
    onnx_model = _staticize_sampling_shapes(onnx_model)
    onnx_model = _rewrite_sampling_argmax_indices(onnx_model, num_views)
    onnx_model = _convert_gather_elements_to_mdc(onnx_model)
    onnx_model = _flatten_sampling_matmuls(
        onnx_model, batch, num_frames, num_views, num_queries,
        num_groups * num_points)
    mdc_custom_ops = {'GridSample3d', 'MDCGatherElements'}
    if any(node.op_type in mdc_custom_ops for node in onnx_model.graph.node):
        # MDC custom operators are not part of the standard ONNX schema.
        print('Skipping standard ONNX checker for MDC custom operator nodes')
    else:
        onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(output_path))
    print('ONNX export complete:', output_path)


if __name__ == '__main__':
    main()
