# OPUS-V2 Pipeline 与模型结构分析

本文以 `configs/opusv2_nusc-occ3d/opusv2-t_r50_704x256_8f_nusc-occ3d_50e.py` 为准，说明相机单模态的 Occ3D 训练与推理路径。OPUS-V2 不构造稠密 BEV 特征图：它从一组可学习的 3D query 出发，逐层扩展为点集，再仅在这些点落入的体素处预测语义。

```text
nuScenes metadata + 8 时刻 × 6 相机图像                         Occ3D GT
              │                                                    │
              ▼                                                    ▼
┌──────────────────────────────────┐                  ┌─────────────────────┐
│ 数据管线                         │                  │ labels.npz          │
│ sweep 选择、读取、IDA 图像增强   │                  │ semantics + masks   │
│ 更新 ego2img / ego2occ 标定      │                  └──────────┬──────────┘
└────────────────┬─────────────────┘                             │
                 ▼                                               │
  [B, 48, 3, 256, 704] + 多帧投影矩阵                            │
                 │                                               │
                 ▼                                               │
┌────────────────────────────────────────────────────────────────┴──────┐
│ OPUS-V2                                                               │
│ ResNet-50 → FPN（4 个尺度，256 通道） → 稀疏点集 Transformer         │
│                                         │                              │
│              600 个可学习初始 3D query ─┘                              │
│                                         ▼                              │
│  5 个 decoder stage：采样多相机/多时刻特征 → query 交互 → 点集细化   │
│  每层：PFN 聚合落入同一体素的点 → SparseConv3D → 体素类别 logits      │
└──────────────────────────────────────────┬────────────────────────────┘
                                           ▼
                          稀疏 (voxel index, semantic label, score)
```

## 数据与预处理

`NuScenesOcc3DDataset` 从 `nuscenes_infos_*_sweep.pkl` 读取当前帧、相机内外参和同 scene 的 sweep。`LoadMVImageWithSweeps(prev_sweeps_num=7)` 选取当前帧及 7 个历史时刻；每个时刻固定六个相机，模型期望共 48 个视图。训练时历史 sweep 间隔在 4–8 个原始 sweep 间随机采样，缺帧时复制最近可用帧。

```text
训练样本
  6 相机 × [当前 t0, 历史 t-1 ... t-7]
       │
       ├─ LoadOcc3DFromFile: [200, 200, 16] 语义体素；empty = 17
       │                     mask_camera / mask_lidar
       ├─ RandomTransformImage: resize、裁剪、可选水平翻转
       │   └─ 同步左乘 IDA 矩阵到每个 ego2img
       ├─ RandomTransformOcc: x/y 随机翻转 GT 与可见性 mask
       │   └─ 同步更新 ego2occ
       └─ FinalFormatting: 图片转为 [48, 3, 256, 704] tensor，标定留在 img_metas
```

图像的颜色扰动、BGR→RGB、归一化和按 32 对齐均在 `OPUSV2.extract_feat()` 的 GPU 路径执行。默认配置的 `use_grid_mask=False`，但 `img_color_aug=True`。测试时图像几何变换是确定性的；单卡在线评测只立即读入当前 6 张图像，模型用文件名缓存历史帧特征（最多保留 15 帧），再重组为 8 时刻特征。多卡评测走离线路径并直接读取全部 48 张图像。

## 图像编码与时空投影

`OPUSV2` 先将 `[B, N, C, H, W]` 展平为 `[B×N, C, H, W]`，经 ResNet-50 和 FPN 得到四个尺度的 256 通道特征；随后恢复为 `[B, 48, 256, H_l, W_l]`。Transformer 使用 `occ2img = ego2img × inverse(ego2occ)`，因此训练时对占据 GT 的翻转也会正确反映到 3D 点投影。

## 稀疏点集 Transformer

```text
初始：600 × 1 个归一化 3D embedding（[0,1]^3） + 零 query feature
                                      │
                                      ▼
┌──────────────────────────────── decoder stage i ────────────────────────────────┐
│  上一层每 query 的 R 个点 → 位置编码                                             │
│        │                                                                         │
│        ├─ Adaptive Spatio-temporal Sampling                                      │
│        │    query feature 预测 4 组 × 4 采样偏移及 4 尺度权重                    │
│        │    3D 采样点投影到 8 时刻 × 6 视图，跨图像/FPN 层取样                  │
│        ├─ Adaptive Mixing：按 query 动态混合采样点及通道                         │
│        ├─ Scale-adaptive self-attention：距离调制 query 间注意力                │
│        ├─ FFN                                                                   │
│        └─ 回归偏移：将 R 点扩展/细化为新的 R' 点，并生成每点 32 维特征           │
└─────────────────────────────────────────────────────────────────────────────────┘
  R'（默认 tiny 配置）: 8 → 16 → 32 → 64 → 128
```

每个 stage 的输入点会 `detach()`，故每一层直接优化其自身的点集；位置仍通过本层的回归分支更新。`OPUSSampling` 的采样范围由同一 query 点集的均值与标准差自适应缩放。最终 600 × 128 = 76,800 个候选点，但不会转化为同等大小的稠密网格。

## CPD：由连续点到离散体素

OPUS-V2 的关键是每层 `Densifier`（Continuous Prediction Discretization）：

```text
连续细化点 [B, Q, R, 3] + 点特征 [B, Q, R, 32]
                 │
                 ▼
按 pc_range [-40,-40,-1, 40,40,5.4] 与 voxel_size 0.4 离散化
                 │                    → 网格 [200,200,16]
                 ▼
相同体素点：相对体素中心坐标 + 相对 query 中心坐标 + 点特征
                 │
                 ▼
两层 PFN（scatter-max 聚合）→ SparseConv3D + SubMConv3D → 17 类 logits
```

体素坐标和 logits 始终是稀疏列表。推理时取最后一层，经过 sigmoid 后保留最大类别分数大于 `score_thr=0.25` 的体素；评测阶段才用 `sparse2dense` 将未预测位置填为 empty（17），计算 mIoU 与 RayIoU。

## 训练目标与入口

训练将 GT 中非 empty 的体素中心转为稀疏点。每层都有两类监督：

- 双向最近邻点损失：预测点与 GT 点之间的 Smooth L1 Chamfer 式匹配；相机不可见或距离匹配较差的 GT 点会提高权重，稀有类别也被重加权。
- 稀疏体素分类：只在 Densifier 实际输出的位置计算 Focal Loss，并使用类别权重。

初始 query 也接受点损失；五层的点与分类损失均被返回。`train.py` 建立 MMCV runner，使用 AdamW、FP16、cosine 学习率和 50 个 epoch；默认从 `pretrain/` 的图像检测骨干加载权重。`val.py` 加载 checkpoint 并调用数据集的 mIoU/RayIoU 评测。

```bash
python train.py --config configs/opusv2_nusc-occ3d/opusv2-t_r50_704x256_8f_nusc-occ3d_50e.py
python val.py --config configs/opusv2_nusc-occ3d/opusv2-t_r50_704x256_8f_nusc-occ3d_50e.py --weights <checkpoint.pth>
```

## 核心代码定位

- 模型封装、GPU 增强与在线缓存：`models/opusv2/opus.py`
- 初始 query、Densifier、损失和稀疏输出：`models/opusv2/opus_head.py`
- decoder、时空采样、动态混合与自注意力：`models/opusv2/opus_transformer.py`
- 图像/占据标注加载和几何增强：`loaders/pipelines/loading.py`、`loaders/pipelines/transforms.py`
- Occ3D 数据集和 mIoU/RayIoU 评测：`loaders/nuscenes_occ3d_dataset.py`
