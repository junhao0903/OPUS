# OPUS-V2 Query、采样与时序机制要点

本文基于默认 Occ3D-T 配置：8 帧、6 相机、600 个 query、5 个 decoder stage、`embed_dims=256`。

## 1. 核心定位

OPUS-V2 是面向 occupancy 的稀疏点集模型。它不构造完整稠密 BEV，而是让固定数量的 query 在三维空间中逐层生成更密的点簇；点簇离散化并聚合后，才输出稀疏体素语义。一个 query 不是对象实例，也不具有显式 ID 或点序拓扑；它表示一个可自适应覆盖局部占据表面的几何—语义单元。

```text
query feature（局部视觉/语义状态） + query points（局部 3D 几何）
                         ↓
多帧图像取证 → 更新 feature → 细化点簇 → 稀疏体素语义
```

## 2. Query 初始化与逐层点集细化

`init_points` 是 `[600, 3]` 的可学习 embedding，模型创建时以 `Uniform(0, 1)` 初始化；训练后坐标随 checkpoint 固化，推理时不会重新随机采样。归一化坐标会映射到实际点云范围（默认 `[-40, -40, -1, 40, 40, 5.4]` 米）。`query_feat` 初始为零，形状始终为 `[B, 600, 256]`。

| 阶段 | 点集形状 `[B, Q, R, 3]` | 每 query 点数 R |
| --- | --- | ---: |
| 初始 | `[B, 600, 1, 3]` | 1 |
| Stage 1–5 输出 | `[B, 600, R, 3]` | 8 / 16 / 32 / 64 / 128 |

点数增长不是逐点复制：每层先对输入点簇取米制坐标均值作为局部中心，再由 `reg_branch` 从 256 维 feature 一次回归 `R' × 3` 偏移，生成新点集。每层输入点坐标会 `detach()`，因此每一层获得直接几何监督，避免坐标梯度跨层不稳定传播；feature 仍连续传递。

## 3. Decoder Layer：算子与职责

```text
points → Position Encoder → OPUSSampling → Adaptive Mixing
       → 距离调制 Self-Attention → FFN
       ├─ reg_branch：生成 R' 个细化点
       └─ cls_branch：生成 R' 个 32 维点特征
```

- **Position Encoder**：编码整个 `R × 3` 点簇几何并注入 query feature。
- **OPUSSampling**：从多帧多视图 FPN 特征取与当前局部 3D 区域有关的证据。
- **Adaptive Mixing**：由 query feature 动态产生通道混合矩阵和时空采样点混合矩阵，将采样结果残差写回 query。
- **Scale-adaptive Self-Attention**：按 query 点簇中心距离生成 attention bias，使 query 在空间关系约束下交换上下文。
- **FFN / 分支**：提炼 feature；输出下一层几何及供 Densifier 使用的点特征。最终类别不是 `cls_branch` 直接输出，而由 Densifier 在体素层预测。

`query_points` 除了生成 sampling keypoints，还用于位置编码、自注意力距离计算、下一层点簇中心、体素离散化与点集损失。

## 4. Sampling Keypoints 与时序融合

每层每个 query 的 sampling keypoint 数量与 R 无关，固定为：

```text
num_groups=4 × num_points=4 = 16 个 3D keypoints / query / frame
8 帧 → 128 个时空采样位置 / query / decoder layer
```

生成公式（米制空间）为：

```text
sampling_point = query_center
               + sampling_prototype × query_scale
               + feature-conditioned sampling_offset
```

其中 center 和 scale 分别为当前点簇的均值与标准差；第一层只有一个点，scale 固定为 1。prototype 是全局学习参数，offset 由当前 query feature 预测。它们有显式 xyz 坐标并通过各帧 `occ2img` 投影至二维图像，单位与点云坐标一致（米）。但这些点没有 box/polyline 一样的实例语义或拓扑顺序。

时序融合不是 backbone 内的融合：48 张图独立经 ResNet/FPN 编码；同一 3D sampling point 再投影到 8 个时刻的图像。输出特征形状为 `[B, Q, G, T×P, C/G]`，Adaptive Mixing 在 `T×P` 维度混合时空证据。它没有显式 timestamp embedding、实例 memory 或动态物体运动补偿；对静态场景有效，对高速动态物体主要依赖网络抑制历史不一致证据。单卡在线模式的图像 feature 缓存仅避免重复 backbone 推理，不是语义 memory。

## 5. Densifier 与训练损失

每层输出点及 32 维点特征经 Densifier 处理：点按 `0.4 m` 体素离散化；同体素内使用 PFN/scatter-max 聚合；随后 SparseConv3D 输出 17 类稀疏体素 logits。推理只保留最终层中分数超过阈值的体素。

训练具有所有 stage 的深度监督：

```text
L = 初始点集损失 + Σ_stage（点集损失 + 稀疏体素分类损失）
```

- **点集损失**：预测点与所有非空 GT voxel 中心做双向最近邻匹配，采用 Smooth L1 的 Chamfer 式双向损失；相机可见但覆盖不佳的点及稀有类会被重加权。
- **分类损失**：只在 Densifier 激活的稀疏体素位置查 GT 标签，采用 sigmoid Focal Loss；empty（标签 17）作为背景处理。

## 6. 与 SparseDrive 的关系

二者都属于“几何 token + feature → 投影图像采样 → 更新 token”的稀疏感知范式，但 token 的语义粒度不同。

| 维度 | SparseDrive | OPUS-V2 |
| --- | --- | --- |
| token | object/map instance | 无拓扑的 occupancy 点簇 query |
| 几何 | box / map polyline 等结构化 anchor | 中心、尺度、连续点集 |
| 图像聚合 | DFA 显式加权多相机、尺度、keypoint | 每点选择至多一个有效相机；再做动态时空/通道 mixing |
| 时间 | instance anchor/feature memory queue | 固定多帧图像重采样，无 query identity memory |
| 输出 | 检测、跟踪、地图、运动、规划 | 稀疏语义 occupancy |

SparseDrive 的结构化 anchor、跨帧 identity 与 memory 更适合动态对象和规划；OPUS-V2 的无拓扑点簇更适合任意表面和非实例化占据。DFA 可接入 OPUS-V2：保留 OPUS 的 keypoint 生成，用 DFA 做多相机/多尺度融合，再保留 Adaptive Mixing 做时间/点/通道重组。收益是更充分的多视角利用；代价是算力、标定敏感性与动态物体跨帧不一致风险上升。

## 7. Query Memory 的可行方向

不能直接复用 SparseDrive 的 instance memory，因为 OPUS query 没有实例身份。更适合的两种方案：

1. **Query memory**：缓存 final-stage `query_feat`、点簇中心、尺度及时间戳；将中心通过相对 ego pose 对齐后，仅与空间邻近的当前 query 做带时间衰减和静态性门控的 cross-attention。
2. **Sparse voxel memory**：缓存 Densifier 的稀疏体素 feature，ego-motion 对齐后与当前稀疏体素融合；这更贴合 occupancy 输出。

二者可提升遮挡恢复、长时静态一致性和在线效率，但必须处理动态物体残影、错误累积、缓存失效、场景切换以及训练—推理时序长度不一致。
