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
    view_index = torch.round(sampling_locations[..., 2] * (N - 1)).long()
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
                   + y_indices.clamp(0, H - 1).long() * W
                   + x_indices.clamp(0, W - 1).long())

        # 一次 gather 读取四个邻点；索引沿通道维扩展，使每个通道取同一组坐标。
        indices = indices.reshape(B, 1, Q * P * 4).expand(B, C, Q * P * 4)
        values = torch.gather(feat, 2, indices).reshape(B, C, Q, P, 4)

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
        onnx.save(onnx_model, str(output_path))
    print('ONNX export complete:', output_path)


if __name__ == '__main__':
    main()
