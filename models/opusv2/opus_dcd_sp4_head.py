import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_scatter
from mmcv.runner import force_fp32, BaseModule
from mmcv.ops import knn, Voxelization
from mmcv.cnn import bias_init_with_prob
from mmdet.core import multi_apply
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
# from spconv.pytorch import (SparseConvTensor, SparseConv3d, SubMConv3d,
#                             SparseSequential)
from ..bbox.utils import decode_points, encode_points
import numpy as np


# PFNLayer without torch_scatter
class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                #  last_layer=False
                 ):
        super().__init__()
        
        # self.last_vfe = last_layer
        self.use_norm = use_norm
        # if not self.last_vfe:
        #     out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        
        self.relu = nn.ReLU()

    def forward(self, inputs):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        # x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        # if self.last_vfe:
        #     return x_max
        # else:
        #     return torch.cat([x, x_max[unq_inv, :]], dim=1)

        return x


class Densifier(BaseModule):

    def __init__(self, in_channels, pfn_channels, num_classes, pc_range, voxel_num):
        super().__init__()
        self.pfn_layers = nn.ModuleList()
        for i, pfn_channel in enumerate(pfn_channels):
            # last_layer = (i == len(pfn_channels) - 1)
            self.pfn_layers.append(
                PFNLayer(
                    in_channels=in_channels + 6 if i == 0 else pfn_channel,
                    out_channels=pfn_channel,
                    use_norm=True,
                    # last_layer=last_layer
                )
            )

        self.cls_branch = nn.Linear(pfn_channels[-1], num_classes, bias=True)
        self.pad_branch = nn.Sequential(
            nn.Linear(pfn_channels[-1], 18, bias=True),
            nn.ReLU())

        self.register_buffer('pc_range', pc_range)
        self.register_buffer('voxel_num', voxel_num)
        self.scale_xyz = self.voxel_num[0] * self.voxel_num[1] * self.voxel_num[2]
        self.scale_yz = self.voxel_num[1] * self.voxel_num[2]
        self.scale_z = self.voxel_num[2]

    def init_weights(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch.bias, bias_init)
        nn.init.constant_(self.pad_branch[0].bias, 1)

    def discretize(self, pts, clip=True):
        voxel_num = self.voxel_num.to(pts.dtype)
        loc = torch.floor(pts * voxel_num)
        if clip:
            loc[..., 0] = loc[..., 0].clamp(0, self.voxel_num[0] - 1)
            loc[..., 1] = loc[..., 1].clamp(0, self.voxel_num[1] - 1)
            loc[..., 2] = loc[..., 2].clamp(0, self.voxel_num[2] - 1)
        
        centers = decode_points((loc + 0.5) / voxel_num, self.pc_range)
        return centers

    def forward(self, pts_feats, refine_pts):
        if pts_feats is None or refine_pts is None:
            return None, None

        B, Q, P, _ = refine_pts.shape
        decode_refine_pts = decode_points(refine_pts, self.pc_range)

        # calculate voxel coordinates
        # batch_index = torch.arange(B, device=pts_feats.device).view(B, 1, 1, 1).expand(B, Q, P, 1)
        coor_centers = self.discretize(refine_pts.detach())  # [B, Q, P, 3]
        # coors = torch.cat([batch_index, coors], dim=-1)
        # coors_flat = coors[..., 0] * self.scale_xyz + coors[..., 1] * self.scale_yz + \
        #     coors[..., 2] * self.scale_z + coors[..., 3]
        
        # calculate voxel features
        centers = decode_refine_pts.mean(dim=2, keepdim=True)
        pts_feats = torch.cat([decode_refine_pts-coor_centers, decode_refine_pts-centers, pts_feats], dim=-1)
        pts_feats = pts_feats.reshape(B * Q * P, -1)
        # coors_flat = coors_flat.reshape(B * Q * P)
        
        for pfn_layer in self.pfn_layers:
            pts_feats = pfn_layer(pts_feats)    # [B, Q, P, pfn_channels[-1]]

        cls_scores = self.cls_branch(pts_feats) # [B, Q, P, num_class]
        pad_scores = self.pad_branch(pts_feats) # [B, Q, P, 18]

        return cls_scores, pad_scores


@HEADS.register_module()
class OPUSV2DCDSP4Head(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 transformer=None,
                 pc_range=[],
                 empty_label=17,
                 pfn_channels=[64, 64],
                 voxel_size=[],
                 train_cfg=dict(),
                 test_cfg=dict(max_per_img=100),
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 loss_pts=dict(type='L1Loss'),
                 init_cfg=None,
                 topk=5,
                 **kwargs):
        super().__init__(init_cfg)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.empty_label = empty_label
        self.loss_cls = build_loss(loss_cls)
        self.loss_pts = build_loss(loss_pts)
        self.transformer = build_transformer(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims

        # prepare scene
        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()
        pad_topk = torch.tensor(topk, dtype=torch.int)
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)
        self.register_buffer('pad_topk', pad_topk)

        self.scale_xyz = self.voxel_num[0] * self.voxel_num[1] * self.voxel_num[2]
        self.scale_yz = self.voxel_num[1] * self.voxel_num[2]
        self.scale_z = self.voxel_num[2]

        pad_xyz = torch.tensor([
                    # [-1, -1, -1],
                    [-1, 0, -1],
                    # [-1, 1, -1],
                    [0, -1, -1],
                    [0, 0, -1],
                    [0, 1, -1],
                    # [1, -1, -1],
                    [1, 0, -1],
                    # [1, 1, -1],
                    [-1, -1, 0],
                    [-1, 0, 0],
                    [-1, 1, 0],
                    [0, -1, 0],
                    # [0, 0, 0],
                    [0, 1, 0],
                    [1, -1, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    # [-1, -1, 1],
                    [-1, 0, 1],
                    # [-1, 1, 1],
                    [0, -1, 1],
                    [0, 0, 1],
                    [0, 1, 1],
                    # [1, -1, 1],
                    [1, 0, 1],
                    # [1, 1, 1],
                ], dtype=torch.float32).unsqueeze(0)  # [1, 18, 3]
        pad_xyz = pad_xyz / self.voxel_num
        self.register_buffer('pad_xyz', pad_xyz)

        # build densifying network
        self.densifiers = nn.ModuleList()
        for _ in range(self.transformer.num_layers):
            self.densifiers.append(
                Densifier(
                    in_channels=self.transformer.num_pt_channels,
                    pfn_channels=pfn_channels,
                    num_classes=num_classes,
                    pc_range=self.pc_range,
                    voxel_num=self.voxel_num
                )
            )

        self._init_layers()

    def _init_layers(self):
        self.init_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.init_points.weight, 0, 1)

    def init_weights(self):
        self.transformer.init_weights()
        for densifier in self.densifiers:
            densifier.init_weights()

    def forward(self, mlvl_feats, img_metas):
        B, Q, = mlvl_feats[0].shape[0], self.num_query
        init_points = self.init_points.weight[None, :, None, :].repeat(B, 1, 1, 1)
        query_feat = init_points.new_zeros(B, Q, self.embed_dims)

        pt_feats, refine_pts = self.transformer(
            init_points, query_feat, mlvl_feats, img_metas=img_metas)   # list of [B, Q, P_(n+1), num_pt_channels] [B, Q, P_(n+1), 3]

        cls_scores_list = []
        pad_scores_list = []
        for i, densifier in enumerate(self.densifiers):
            cls_scores, pad_scores = densifier(pt_feats[i], refine_pts[i])
            cls_scores_list.append(cls_scores)
            pad_scores_list.append(pad_scores)

        return dict(init_points=init_points,
                    all_refine_pts=refine_pts,
                    all_cls_scores=cls_scores_list,
                    all_pad_scores=pad_scores_list)
       

    def discretize(self, pts, clip=True):
        voxel_num = self.voxel_num.to(pts.dtype)
        loc = torch.floor(pts * voxel_num)
        if clip:
            loc[..., 0] = loc[..., 0].clamp(0, self.voxel_num[0] - 1)
            loc[..., 1] = loc[..., 1].clamp(0, self.voxel_num[1] - 1)
            loc[..., 2] = loc[..., 2].clamp(0, self.voxel_num[2] - 1)
        
        return loc.long()

    def get_dis_weight(self, pts):
        max_dist = torch.sqrt(
            self.scene_size[0] ** 2 + self.scene_size[1] ** 2)
        centers = (self.pc_range[:3] + self.pc_range[3:]) / 2
        dist = (pts - centers[None, ...])[..., :2]
        dist = torch.norm(dist, dim=-1)
        return dist / max_dist + 1

    def get_unique_sum_scores(self, all_refine_pts, all_cls_scores, all_pad_scores):
        new_all_cls_scores = []
        new_all_unique_coors = []
        
        for i in range(len(all_refine_pts)):
            if all_cls_scores[i] is None:
                new_all_cls_scores.append(None)
                new_all_unique_coors.append(None)
            # else:   # just for test
            #     B, Q, P, _ =  all_refine_pts[i].shape
            #     refine_pts = all_refine_pts[i].reshape(-1, 1, 3)                    # [B*Q*P_n, 1, 3]

            #     pad_scores = all_pad_scores[i].reshape(-1, 18)                      # [B*Q*P_n, 18]
            #     topk_pad_scores, topk_idx = torch.topk(pad_scores, k=self.pad_topk, dim=-1, largest=False)     # [B*Q*P_n, topk], lower is better
            #     first_idx = torch.arange(B*Q*P, device=pad_scores.device).view(B*Q*P, 1, 1)
            #     last_idx = torch.arange(3, device=pad_scores.device).view(1, 1, 3)
            #     topk_idx = topk_idx.view(B*Q*P, self.pad_topk, 1)                   # [B*Q*P_n, topk, 1]
            #     exp_pad_xyz = self.pad_xyz.expand(B*Q*P, 18, 3)                     # [B*Q*P_n, 18, 3]
            #     topk_pad_xyz = exp_pad_xyz[first_idx, topk_idx, last_idx]           # [B*Q*P_n, topk, 3]
            #     topk_pad_scores = topk_pad_scores.view(B*Q*P, self.pad_topk, 1)     # [B*Q*P_n, topk, 1]

            #     # refine_pts = decode_points(refine_pts, self.pc_range)
            #     cls_scores = all_cls_scores[i].reshape(-1, 1, self.num_classes)     # [B*Q*P_n, 1, num_cls]
                
            #     pad_cls_scores = cls_scores * topk_pad_scores                       # [B*Q*P_n, topk, num_cls]
            #     pad_refine_pts = refine_pts + topk_pad_xyz                          # [B*Q*P_n, topk, 3]
            #     # pad_refine_pts = encode_points(pad_refine_pts, self.pc_range)
            #     new_refine_pts = torch.cat([refine_pts, pad_refine_pts], dim=1).reshape(-1, 3)                   # [B*Q*P_n*(topk+1), 3]
            #     new_cls_scores = torch.cat([cls_scores, pad_cls_scores], dim=1).reshape(-1, self.num_classes)    # [B*Q*P_n*(topk+1), num_cls]
            #     coors = self.discretize(new_refine_pts.detach()).reshape(-1, 3)     # [B*Q*P_n*(topk+1), 3]

            #     batch_index = torch.arange(B, device=refine_pts.device).view(B, 1, 1, 1, 1).expand(B, Q, P, self.pad_topk + 1, 1).reshape(-1, 1)
            #     coors = torch.cat([batch_index, coors], dim=-1)                     # [B*Q*P_n*(topk+1), 4]
            #     coors_flat = coors[..., 0] * self.scale_xyz + coors[..., 1] * self.scale_yz + \
            #         coors[..., 2] * self.scale_z + coors[..., 3]                    # [B*Q*P_n*(topk+1)]

            #     unique_coors_flat, inverse_idx, counts = torch.unique(coors_flat, return_inverse=True, return_counts=True)          # [unique_idx_num]
            #     counts = counts.unsqueeze(1)
            #     U = unique_coors_flat.shape[0]
            #     unique_cls_scores = torch.zeros(U, self.num_classes, dtype=cls_scores.dtype, device=cls_scores.device)
            #     unique_cls_scores.index_add_(dim=0, index=inverse_idx, source=new_cls_scofres)           # [unique_idx_num, num_cls]
            #     unique_cls_scores = unique_cls_scores / counts

            #     new_all_cls_scores.append(unique_cls_scores)
            #     new_all_unique_coors.append(unique_coors_flat)
            else:
                B, Q, P, _ =  all_refine_pts[i].shape
                refine_pts = all_refine_pts[i].reshape(-1, 1, 3)                    # [B*Q*P_n, 1, 3]
                # refine_pts = decode_points(refine_pts, self.pc_range)
                cls_scores = all_cls_scores[i].reshape(-1, 1, self.num_classes)     # [B*Q*P_n, 1, num_cls]
                pad_scores = all_pad_scores[i].reshape(-1, 18, 1)                    # [B*Q*P_n, 18, 1]
                pad_cls_scores = cls_scores * pad_scores                            # [B*Q*P_n, 18, num_cls]
                pad_refine_pts = refine_pts + self.pad_xyz                          # [B*Q*P_n, 18, 3]
                # pad_refine_pts = encode_points(pad_refine_pts, self.pc_range)
                new_refine_pts = torch.cat([refine_pts, pad_refine_pts], dim=1).reshape(-1, 3)                   # [B*Q*P_n*19, 3]
                new_cls_scores = torch.cat([cls_scores, pad_cls_scores], dim=1).reshape(-1, self.num_classes)    # [B*Q*P_n*19, num_cls]
                coors = self.discretize(new_refine_pts.detach()).reshape(-1, 3)     # [B*Q*P_n*19, 3]

                batch_index = torch.arange(B, device=refine_pts.device).view(B, 1, 1, 1, 1).expand(B, Q, P, 19, 1).reshape(-1, 1)
                coors = torch.cat([batch_index, coors], dim=-1)                     # [B*Q*P_n*19, 4]
                coors_flat = coors[..., 0] * self.scale_xyz + coors[..., 1] * self.scale_yz + \
                    coors[..., 2] * self.scale_z + coors[..., 3]                    # [B*Q*P_n*19]

                # unique_coors_flat, inverse_idx = torch.unique(coors_flat, return_inverse=True)          # [unique_idx_num]
                # counts = torch.bincount(inverse_idx).float().unsqueeze(1)
                unique_coors_flat, inverse_idx, counts = torch.unique(coors_flat, return_inverse=True, return_counts=True)          # [unique_idx_num]
                counts = counts.unsqueeze(1)

                U = unique_coors_flat.shape[0]
                unique_cls_scores = torch.zeros(U, self.num_classes, dtype=cls_scores.dtype, device=cls_scores.device)
                unique_cls_scores.index_add_(dim=0, index=inverse_idx, source=new_cls_scores)           # [unique_idx_num, num_cls]
                unique_cls_scores = unique_cls_scores / counts
            
                new_all_cls_scores.append(unique_cls_scores)
                new_all_unique_coors.append(unique_coors_flat)
        return new_all_cls_scores, new_all_unique_coors


    @torch.no_grad()
    def _get_regression_target_single(self, refine_pts, gt_points, gt_masks, gt_labels):
        # knn to apply Chamfer distance
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])     # [1, 1, gt_num_points]
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()
        gt_paired_num = torch.bincount(gt_paired_idx)[gt_paired_idx]
        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])   # [1, 1, refine_num_points]
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze().long()
        pred_paired_num = torch.bincount(pred_paired_idx)[pred_paired_idx]
        gt_paired_pts = refine_pts[gt_paired_idx]

        # gt side weights
        empty_dist_thr = self.train_cfg.get('empty_dist_thr', 0.2)
        empty_weights = self.train_cfg.get('empty_weights', 5)

        gt_pts_weights = refine_pts.new_ones(gt_paired_pts.shape[0])
        dist = torch.norm(gt_points - gt_paired_pts, dim=-1)
        mask = (dist > empty_dist_thr) & gt_masks
        gt_pts_weights[mask] = empty_weights

        rare_classes = self.train_cfg.get('rare_classes', [0, 2, 5, 8])
        rare_weights = self.train_cfg.get('rare_weights', 10)
        for cls_idx in rare_classes:
            mask = (gt_labels == cls_idx) & gt_masks
            gt_pts_weights[mask] = gt_pts_weights[mask].clamp(min=rare_weights)
        
        gt_pts_weights = gt_pts_weights / gt_paired_num.float()
        gt_avg_factor = (1 / gt_paired_num.float()).sum()
        
        # prediction side weights
        pred_pts_weights = refine_pts.new_ones(refine_pts.shape[0])
        pred_pts_weights = pred_pts_weights / pred_paired_num.float()
        pred_avg_factor = (1 / pred_paired_num.float()).sum()

        return (gt_paired_idx, pred_paired_idx, gt_pts_weights, pred_pts_weights,
                gt_avg_factor, pred_avg_factor)
    
    def get_targets(self):
        # To instantiate the abstract method
        pass

    def loss_single(self,
                    refine_pts,
                    new_cls_scores, 
                    new_unique_coors,
                    gt_points_list,
                    gt_masks_list,
                    gt_labels_list,
                    gt_dense_occ):
        num_imgs = refine_pts.size(0) # B
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        # calculate loss pts
        (gt_paired_idx_list, pred_paired_idx_list, gt_pts_weights, pred_pts_weights,
         gt_avg_factor, pred_avg_factor) = multi_apply(
             self._get_regression_target_single, refine_pts_list, gt_points_list,
             gt_masks_list, gt_labels_list)
        
        gt_paired_pts, pred_paired_pts= [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])

        # concatenate all results from different samples
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)
        pred_gt_weights = torch.cat(pred_pts_weights)

        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=sum(gt_avg_factor))
        loss_pts += self.loss_pts(pred_pts, 
                                  pred_paired_pts,
                                  weight=pred_gt_weights[..., None],
                                  avg_factor=sum(pred_avg_factor))

        # calculate loss cls
        loss_cls = pred_pts.new_tensor(0)
        if (new_cls_scores is not None) and (new_unique_coors is not None):
            gt_dense_occ_flat = gt_dense_occ.reshape(-1)
            gt_labels = gt_dense_occ_flat[new_unique_coors]

            # calculate cls weights
            cls_weights = self.train_cfg.get('cls_weights', [1] * self.num_classes)
            cls_weights = new_cls_scores.new_tensor(cls_weights)
            cls_weights = cls_weights[None, :].expand(new_cls_scores.shape[0], -1)
            avg_factor = (gt_labels != self.empty_label).sum()

            loss_cls += self.loss_cls(new_cls_scores,
                                      gt_labels,
                                      weight=cls_weights,
                                      avg_factor=avg_factor)

        return loss_cls, loss_pts

    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics, mask_camera, preds_dicts):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17
        init_points = preds_dicts['init_points']        # [B, Q, P_0, 3]
        all_refine_pts = preds_dicts['all_refine_pts']  # list of tensor : [B, Q, P_n, 3]

        all_cls_scores = preds_dicts['all_cls_scores']  # list of tensor : [B*Q*P_n, num_cls]
        all_pad_scores = preds_dicts['all_pad_scores']  # list of tensor : [B*Q*P_n, 18]

        voxel_semantics = voxel_semantics.long()        # [B, occ_x, occ_y, occ_z]
        mask_camera = mask_camera.bool()                # [B, occ_x, occ_y, occ_z]


        new_all_cls_scores, new_all_unique_coors = self.get_unique_sum_scores(
            all_refine_pts, all_cls_scores, all_pad_scores)


        num_dec_layers = len(all_refine_pts)
        gt_points_list, gt_masks_list, gt_labels_list = \
            self.get_sparse_voxels(voxel_semantics, mask_camera)
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_dense_occ = [voxel_semantics for _ in range(num_dec_layers)]

        losses_cls, losses_pts = multi_apply(
            self.loss_single, all_refine_pts, new_all_cls_scores, new_all_unique_coors,
            all_gt_points_list, all_gt_masks_list, all_gt_labels_list, all_gt_dense_occ)

        loss_dict = dict()
        # loss of init_points
        if init_points is not None:
            _, init_loss_pts = self.loss_single(
                init_points, None, None, gt_points_list, 
                gt_masks_list, gt_labels_list, voxel_semantics)
            loss_dict['init_loss_pts'] = init_loss_pts
        
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_pts'] = losses_pts[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(losses_cls[:-1], losses_pts[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            num_dec_layer += 1
        return loss_dict

    
    def get_occ(self, pred_dicts, img_metas, rescale=False):
        all_refine_pts = pred_dicts['all_refine_pts']  # list of tensor : [B, Q, P_n, 3]
        all_cls_scores = pred_dicts['all_cls_scores']  # list of tensor : [B*Q*P_n, num_cls]
        all_pad_scores = pred_dicts['all_pad_scores']  # list of tensor : [B*Q*P_n, 18]

        new_all_cls_scores, new_all_unique_coors = self.get_unique_sum_scores(
            all_refine_pts, all_cls_scores, all_pad_scores)
        
        voxels = new_all_unique_coors[-1]       # [unique_idx_num]
        voxels = torch.stack(
            [voxels // self.scale_xyz,
                (voxels % self.scale_xyz) // self.scale_yz,
                (voxels % self.scale_yz) // self.scale_z,
                (voxels % self.scale_z)], dim=1
        )   # # [unique_idx_num, 4]
        cls_scores = new_all_cls_scores[-1].sigmoid()
        batch_size = all_refine_pts[-1].shape[0]
        scores, labels = cls_scores.max(dim=-1)

        mask = scores > self.test_cfg.get('score_thr', 0.25)
        labels = labels[mask]
        voxels = voxels[mask]

        result_list = []
        for i in range(batch_size):
            _labels = labels[voxels[:, 0] == i]
            _voxels = voxels[voxels[:, 0] == i, 1:]
            result_list.append(dict(
                sem_pred=_labels.detach().cpu().numpy(),
                occ_loc=_voxels.detach().cpu().numpy()))

        return result_list
    
    def get_sparse_voxels(self, voxel_semantics, mask_camera):
        B, W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()

        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, W, Z)
        coors = torch.stack([xx, yy, zz], dim=-1) # actual space

        gt_points, gt_masks, gt_labels = [], [], []
        for i in range(B):
            mask = voxel_semantics[i] != self.empty_label
            gt_points.append(coors[mask])
            gt_masks.append(mask_camera[i][mask]) # camera mask and not empty
            gt_labels.append(voxel_semantics[i][mask])
        
        return gt_points, gt_masks, gt_labels
