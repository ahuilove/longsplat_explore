# Code Understanding Notes

## train.py: GaussianModel 初始化

调试位置：`train.py` 中 `training(...)` 函数刚开始的位置。

相关代码：

```python
gaussians = GaussianModel(
    dataset.feat_dim,
    dataset.n_offsets,
    dataset.voxel_size,
    dataset.update_depth,
    dataset.update_init_factor,
    dataset.update_hierachy_factor,
    dataset.use_feat_bank,
    dataset.appearance_dim,
    dataset.ratio,
    dataset.add_opacity_dist,
    dataset.add_cov_dist,
    dataset.add_color_dist,
)
```

这一步是在创建 LongSplat 的核心模型对象 `GaussianModel`。此时只是确定模型结构和超参数，还没有真正创建 anchor 点。

### 参数含义

`dataset.feat_dim`

每个 anchor 的可学习特征维度。默认一般是 `32`。每个 anchor 会保存一个 `feat_dim` 维 latent feature，后续用于 MLP 预测 opacity、color、scale 和 rotation。

`dataset.n_offsets`

每个 anchor 周围挂载多少个 offset Gaussian。LongSplat 不是一个 anchor 直接对应一个 Gaussian，而是：

```text
anchor -> n_offsets 个局部 offset Gaussian
```

例如 `n_offsets=10` 时，理论候选 Gaussian 数量约为：

```text
anchor 数量 * 10
```

但实际渲染时，opacity MLP 会预测每个 offset 的 opacity，只有 `neural_opacity > 0` 的 offset 会被保留下来参与 rasterization。

`dataset.voxel_size`

初始化 anchor 的体素大小。MASt3R 输出的点云会先经过 voxel / octree 采样，再变成 LongSplat 的 anchor。`voxel_size` 越小，anchor 越密，细节潜力更高，但显存和训练开销也更大。

如果 `voxel_size <= 0`，代码会根据点云最近邻距离自动估计一个 voxel size。

`dataset.update_depth`

octree / 层级采样的深度。它会影响初始化和后续 densification 的层级结构。值越大，高密度区域可能被细分得更细，anchor 数量更多。

`dataset.update_init_factor`

anchor densification 相关的初始化控制参数。它不会在 `GaussianModel.__init__` 中直接生成点，而是保存到模型里，后续在动态增长 anchor 时参与控制增长尺度或层级。

`dataset.update_hierachy_factor`

层级式 anchor 更新的比例控制参数。变量名里 `hierachy` 应该是 `hierarchy` 的拼写错误。它和 `update_depth`、`update_init_factor` 一起控制 LongSplat 训练过程中 anchor 的层级式增长。

`dataset.use_feat_bank`

是否启用 feature bank。如果为 `True`，模型会额外创建 `mlp_feature_bank`，根据观察方向 `ob_view` 和距离 `ob_dist` 动态融合不同尺度的 anchor feature。默认通常是 `False`。

`dataset.appearance_dim`

每个相机的 appearance embedding 维度。如果大于 `0`，模型会为每个相机创建一个 appearance embedding，并把它拼到 color MLP 的输入中，用来建模不同帧之间的曝光、白平衡或光照差异。默认通常是 `0`，表示不启用。

`dataset.ratio`

初始化点云的采样比例。在 `create_from_pcd(...)` 中使用：

```python
points = pcd.points[::self.ratio]
```

`ratio=1` 表示使用全部点；`ratio=2` 表示每隔一个点取一个，大约使用一半点。

`dataset.add_opacity_dist`

是否把相机到 anchor 的距离 `ob_dist` 加入 opacity MLP 的输入。启用后，opacity 预测会显式感知视距。

`dataset.add_cov_dist`

是否把相机到 anchor 的距离 `ob_dist` 加入 covariance MLP 的输入。启用后，Gaussian 的 scale 和 rotation 预测会显式感知视距。

`dataset.add_color_dist`

是否把相机到 anchor 的距离 `ob_dist` 加入 color MLP 的输入。启用后，颜色预测会显式感知视距。

## scene/__init__.py: 写入 MASt3R 初始化位姿和深度

调试位置：`matcher.global_align(...)` 返回之后，对 `world2cam` 的循环。

这段代码处理的是前 `init_frame_num` 个初始化帧。它把 MASt3R 全局对齐得到的初始相机位姿、深度图和相邻帧匹配点写入 `Camera` 对象。

核心逻辑：

```text
for 每个初始化帧:
    1. 从 world2cam 取出 R/T，并写入当前 Camera
    2. 用 MASt3R 对相邻两帧做 matching，保存 kp0/kp1
    3. 把 global_align 得到的 depth_map resize 到当前训练图像尺寸
    4. 标记当前帧 is_registered=True
```

关键变量：

`world2cam`

MASt3R 返回的初始化帧外参，形状通常是 `(init_frame_num, 4, 4)`。每个 `Rt` 是一帧的 world-to-camera 矩阵。

`R = Rt[:3, :3].t()`

取旋转矩阵并转置后存入 `Camera`。这里转置是因为项目沿用 3DGS/COLMAP 的内部表示约定。

`T = Rt[:3, 3]`

取 world-to-camera 平移向量。

`cur_viewpoint_cam.update_RT(R, T)`

把 MASt3R 估计出的初始位姿写入当前训练相机。

`matcher._forward(...)`

对相邻两帧做 MASt3R 两视图匹配，主要保存：

```text
kp0: 第一张输入图中的匹配点
kp1: 第二张输入图中的对应匹配点
```

第 0 帧没有前一帧，所以用第 0 帧和第 1 帧匹配；后续帧都用前一帧和当前帧匹配。保存这些匹配点是为了后续训练里的 2D correspondence loss。

`cur_viewpoint_cam.depth_map`

来自 `global_align(...)` 返回的 `depth_maps[i]`。先从 numpy 转成 CUDA tensor，再用 `F.interpolate` resize 到当前 `Camera` 的训练图像尺寸。后续会作为 depth loss 的监督。

`cur_viewpoint_cam.is_registered = True`

表示这些初始化帧已经有可用位姿。后续 `train.py` 的增量训练只会对未注册的新帧做 PnP 初始化。

后续用途：

```text
R/T       -> 初始化相机位姿，用于渲染和 pose optimization
kp0/kp1   -> 2D correspondence loss
depth_map -> depth loss
is_registered -> 控制是否需要对该帧重新做 PnP 注册
```

## MASt3R global_align 的具体流程

调试位置：`utils/mast3r_utils.py` 中的 `Mast3rMatcher.global_align(...)`。

这一步用于从前 `init_frame_num` 张图中估计初始全局场景。它不是让网络一次性输入多张图，而是“先两两预测，再全局优化”。

整体流程：

```text
1. load_images(paths, size=512)
   读取初始化图像，并 resize 到 MASt3R 使用的 512 尺度。

2. make_pairs(images, scene_graph='complete', symmetrize=True)
   构造图像 pair。默认 complete graph 会让每张图和其他图都配对。
   如果有 3 张图，基础 pair 是 0-1、0-2、1-2；
   加上 symmetrize=True 后变成正反方向共 6 个 pair。

3. sparse_global_alignment(...)
   对所有 pair 做 MASt3R 前向和稀疏全局优化。
   这一步内部会先跑 pairwise prediction，再把所有 pair 的匹配/点图/深度合并到一个全局优化问题里。

4. scene.get_dense_pts3d(clean_depth=False)
   从优化后的 MASt3R scene 中取每张图的稠密 3D 点、深度图和置信度。

5. scene.get_im_poses()
   取每张图的 camera-to-world 位姿，并取逆得到 world-to-camera：
   world2cam = inverse(cam2world)

6. scene.get_focals()
   取 MASt3R 估计出的焦距，并计算平均焦距 avg_focal。

7. compute_co_vis_masks(...)
   根据深度、3D 点、相机内外参计算共视关系，用来过滤冗余或不可靠点。

8. 拼接过滤后的 pts3d
   把每张图保留下来的 3D 点拼成一个全局点云，形状变成 (N, 3)。

9. 返回 pts3d, world2cam, depth_maps, avg_focal
```

`sparse_global_alignment(...)` 内部可以理解为：

```text
forward_mast3r:
    对每个 pair 跑 MASt3R 前向，得到描述子、点图、置信度等 pairwise 结果。

prepare_canonical_data / condense_data:
    整理 pairwise 结果，提取全局优化需要的匹配、深度、焦距初值、anchor 等信息。

sparse_scene_optimizer:
    真正做全局优化。优化变量包括每张图的旋转、平移、深度，以及可能的内参。

SparseGA:
    保存优化后的全局结果，并提供 get_dense_pts3d、get_im_poses、get_focals 等接口。
```

返回值含义：

```text
pts3d:
    过滤并拼接后的全局点云，形状是 (N, 3)。
    后续传给 create_from_pcd(...) 初始化 Gaussian anchor。

world2cam:
    初始化帧的 world-to-camera 矩阵，形状通常是 (init_frame_num, 4, 4)。
    后续写入每个初始化 Camera 的 R/T。

depth_maps:
    每个初始化帧一张 MASt3R 深度图。
    后续 resize 到训练图像大小，作为 depth loss 的监督。

avg_focal:
    MASt3R 估计的平均焦距。
    如果没有 COLMAP 内参，会用它更新所有相机的 focal。
```

一句话总结：

```text
global_align = pairwise MASt3R 预测 + 多视图稀疏全局优化，
最终得到 LongSplat 初始化需要的点云、相机位姿、深度图和焦距。
```

## scene/gaussian_model.py: optimizer 要优化的参数

调试位置：`GaussianModel.training_setup(...)` 中构造 `l = [...]` 的位置。

`l` 是传给 Adam optimizer 的参数组列表。它定义了训练时哪些张量/网络会被优化，以及每组参数对应的学习率。

```python
self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
```

虽然 Adam 外层 `lr=0.0`，但每个 param group 都单独设置了自己的 `lr`，实际使用的是各组自己的学习率。

可以把参数分成两类：

```text
1. 显式场景参数:
   _anchor, _offset, _anchor_feat, _opacity, _scaling, _rotation

2. 神经解码器:
   mlp_opacity, mlp_cov, mlp_color
```

显式场景参数：

`_anchor`

anchor 的 3D 坐标，形状约为 `(num_anchors, 3)`。默认 `position_lr_init=0`，所以当前配置下一般不会主动移动 anchor 位置。

`_offset`

每个 anchor 周围的局部 offset，形状约为 `(num_anchors, n_offsets, 3)`。实际 Gaussian 中心近似由：

```text
xyz = anchor + offset * scaling
```

得到。它是重要的几何优化参数。

`_anchor_feat`

每个 anchor 的 latent feature，形状约为 `(num_anchors, feat_dim)`。它不会直接作为颜色，而是输入 MLP，用来预测 opacity、color、scale 和 rotation。

`_opacity`

anchor 级别的 opacity 参数。当前主渲染路径里，最终 opacity 主要由 `mlp_opacity` 预测，因此 `_opacity` 更偏保留参数/辅助参数。

`_scaling`

anchor 的尺度参数，形状约为 `(num_anchors, 6)`。前 3 维影响 offset 的空间范围，后 3 维参与最终 Gaussian scale。

`_rotation`

anchor 级别旋转参数，通常是 quaternion，形状约为 `(num_anchors, 4)`。当前 neural Gaussian 的最终 rotation 主要由 `mlp_cov` 输出；如果 `_rotation.requires_grad=False`，即使放进 optimizer 也不会被更新。

神经解码器：

`mlp_opacity`

根据 `anchor_feat + view direction` 预测每个 offset Gaussian 的 opacity。输出维度是 `n_offsets`。渲染时只保留 `neural_opacity > 0` 的 offset。

`mlp_cov`

预测每个 offset Gaussian 的形状参数。每个 offset 输出 7 维：

```text
3 维 scale + 4 维 rotation
```

因此总输出维度是 `7 * n_offsets`。

`mlp_color`

预测每个 offset Gaussian 的 RGB。每个 offset 输出 3 维颜色，因此总输出维度是 `3 * n_offsets`。最后经过 `Sigmoid`，颜色范围在 `[0, 1]`。

一句话总结：

```text
optimizer 同时优化显式场景表征和 MLP 解码器：
anchor/offset/scaling/feature 决定“在哪里”；
mlp_opacity/mlp_cov/mlp_color 决定“是否可见、长什么形状、是什么颜色”。
```

self.cameras_extent 这是场景的尺度信息

## 相机列表和相机位姿优化

### getTrainCameras(scale) 的含义

`Scene` 里的 `train_cameras` 和 `test_cameras` 不是直接的 list，而是 dict：

```python
self.train_cameras = {}
self.test_cameras = {}
```

初始化时按分辨率尺度保存相机列表：

```python
self.train_cameras[resolution_scale] = cameraList_from_camInfos(...)
self.test_cameras[resolution_scale] = cameraList_from_camInfos(...)
```

默认 `resolution_scales=[1.0]`，所以实际结构类似：

```python
self.train_cameras = {
    1.0: [Camera0, Camera1, Camera2, ...]
}
```

因此：

```python
scene.getTrainCameras()
```

等价于：

```python
scene.train_cameras[1.0]
```

返回的是默认尺度下的训练相机列表。后续的：

```python
scene.getTrainCameras()[0:end_view_id]
```

是在对这个列表做切片。

### cam_rot_delta 和 cam_trans_delta 来源

每个 `Camera` 在构造时都会创建两个可学习的位姿增量：

```python
self.cam_rot_delta = nn.Parameter(torch.zeros(3, requires_grad=True, device=data_device))
self.cam_trans_delta = nn.Parameter(torch.zeros(3, requires_grad=True, device=data_device))
```

含义：

```text
cam_rot_delta   -> 3 维旋转增量
cam_trans_delta -> 3 维平移增量
```

代码没有直接把 `camera.R` 和 `camera.T` 放进 optimizer，而是优化这两个小增量。


### 位姿增量如何影响渲染

渲染时，rasterizer 会接收当前相机的 pose delta：

```python
theta = viewpoint_camera.cam_rot_delta
rho   = viewpoint_camera.cam_trans_delta
```

这样 image loss 反传时，梯度可以传到 `cam_rot_delta` 和 `cam_trans_delta`。

### update_pose 如何写回 R/T

每次 `pose_optimizer.step()` 后，训练代码会调用：

```python
update_pose(viewpoint_cam)
```

核心逻辑：

```text
1. 把 cam_trans_delta 和 cam_rot_delta 拼成 tau
2. 用 SE3_exp(tau) 把小增量转成 4x4 位姿变换
3. 左乘到当前 world-to-camera 位姿上
4. 得到新的 R/T，并写回 camera
5. 清零 cam_rot_delta 和 cam_trans_delta
```

所以相机优化是增量式的：

```text
优化 delta -> 合并进 R/T -> 清零 delta -> 下一轮继续优化新的 delta
```

## train.py 整体流程示例

下面用一组具体数字理解 `train.py` 的完整流程。

假设：

```text
num_views = 10
init_frame_num = 3

init_iteraion = 3000
pose_iteration = 200
local_iter = 400
global_iter = 900
post_iter = 20000

update_interval = 100
start_stat = 0
update_from = 0
```

帧编号：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

### 1. 初始化阶段

先用 MASt3R 初始化前 3 帧：

```text
frames [0,1,2]
```

`global_align(...)` 输出初始点云、初始相机位姿和深度图：

```text
pts3d      -> 初始化 Gaussian anchor
world2cam  -> 初始化 frames 0,1,2 的相机位姿
depth_maps -> 初始化 depth loss 的深度监督
```

然后进行初始优化：

```text
Init Optimization: iteration 1 ~ 3000
训练视角: frames [0,1,2]
```

densification 时机：

```text
统计: 1 ~ 2999
adjust_anchor: 100,200,300,...,2900
清理统计缓存: 3000
```

初始化结束后：

```text
end_view_id = 4
下一轮准备加入 frame 3
```

### 2. 增量 while 循环

主循环负责一帧一帧加入新图像：

```text
while start_view_id < num_views:
    注册新帧 pose
    local optimization
    global optimization
    根据可见 Gaussian overlap 更新窗口
```

关键变量：

```text
start_view_id -> local window 起点
end_view_id   -> 当前已纳入训练范围的右边界
num_views     -> 总帧数
```

以第 1 轮 while 为例：

```text
start_view_id = 0
end_view_id = 4
当前要加入的新帧 = end_view_id - 1 = frame 3
```

如果 `frame 3.is_registered == False`，会执行：

```text
1. 渲染前一帧 frame 2 的 depth
2. 用 MASt3R 匹配 frame 2 和 frame 3，得到 kp0/kp1
3. 用 frame 2 的 rendered_depth 把 kp0 反投影成 3D 点
4. 用 3D 点 + frame 3 的 kp1 做 PnP
5. 用 least_squares 细化 PnP
6. 写入 frame 3 的 R/T
7. 对齐 frame 3 的 depth scale
8. pose refinement 200 iter，只优化 frame 3 的 pose delta
9. densify_occlusion 一次，为新出现区域补 anchor
```

### 3. Local Optimization

新帧注册后先做局部优化。

第 1 轮 while：

```text
local window = scene.getTrainCameras()[0:4]
             = frames [0,1,2,3]
Local Optimization: 400 iter
```

每次迭代随机选窗口内一帧：

```text
render -> loss -> backward -> optimizer step
```

local densification：

```text
统计: 1 ~ 399
adjust_anchor: 100,200,300
清理统计缓存: 400
require_purning = False
```

含义：

```text
local 阶段重点让新帧和附近窗口稳定，通常不强制 prune。
```

### 4. Global Optimization

local 后做全局优化。

第 1 轮 while：

```text
global range = scene.getTrainCameras()[0:4]
             = frames [0,1,2,3]
Global Optimization: 900 iter
```

global densification：

```text
统计: 1 ~ 899
adjust_anchor: 100,200,300,400,500,600,700,800
清理统计缓存: 900
```

含义：

```text
global 阶段把所有已注册帧一起优化，减少累计漂移。
```

### 5. 局部窗口如何变化

每轮 global 后，会比较窗口末端帧和窗口前面帧的可见 Gaussian overlap。

如果：

```text
visibility_ratio < 0.2
并且 end_view_id - start_view_id > 5
```

则：

```text
start_view_id += 1
```

说明最早的窗口帧和当前新帧共视内容太少，local window 可以右移。

然后：

```text
end_view_id += 1
```

准备加入下一帧。

一个可能的窗口变化：

```text
初始化:
    frames [0,1,2]

while 第 1 轮:
    加入 frame 3
    local  = [0,1,2,3]
    global = [0,1,2,3]
    end_view_id -> 5

while 第 2 轮:
    加入 frame 4
    local  = [0,1,2,3,4]
    global = [0,1,2,3,4]
    end_view_id -> 6

while 第 3 轮:
    加入 frame 5
    local  = [0,1,2,3,4,5]
    global = [0,1,2,3,4,5]
    如果 overlap 低，start_view_id: 0 -> 1
    end_view_id -> 7

while 第 4 轮:
    加入 frame 6
    local  = [1,2,3,4,5,6]
    global = [0,1,2,3,4,5,6]
```

关键区别：

```text
Local Optimization 用滑动窗口 [start_view_id:end_view_id]
Global Optimization 用所有已注册帧 [0:end_view_id]
```

### 5.1 新帧 PnP 失败后的 retry

每轮 while 准备注册的新帧是：

```text
frame_id = end_view_id - 1
```

如果 `cv2.solvePnPRansac` 成功，并且 inliers 数量不少于 4：

```text
viewpoint_cam.is_registered = True
```

然后进入 pose refinement、新区域 densification、local/global optimization。

如果 PnP 失败：

```python
viewpoint_cam.is_registered = False
pnp_retry_count[frame_id] += 1
```

如果失败次数还没到 `max_pnp_retries`：

```python
end_view_id -= 1
```

含义是：这次先不把新帧加入训练窗口，退回去继续优化已有帧。while 末尾又会 `end_view_id += 1`，所以下一轮会重新尝试注册同一个新帧。

例子：

```text
当前 end_view_id = 4
准备注册 frame 3

frame 3 PnP 失败:
    end_view_id: 4 -> 3
    本轮 local/global 只训练 frames [0,1,2]

本轮结束:
    end_view_id: 3 -> 4

下一轮:
    再次尝试注册 frame 3
```

这样做的目的：旧模型或上一帧渲染深度可能还不够准，先优化已有帧，下一轮 PnP 可能会更稳定。

如果失败次数达到 `max_pnp_retries`：

```python
viewpoint_cam.is_registered = True
```

相当于强行跳过这个失败帧，避免训练一直卡在同一帧。

注意：当前代码里 PnP 失败后，后面仍会继续使用 `rotation_vector` 和 `translation_vector` 更新相机位姿；如果 OpenCV 返回的值无效，这里可能有风险。更稳妥的实现应该在失败且未达到最大 retry 次数时直接跳过后续 pose update 和 densification。

### 5.2 新帧注册后的新区域 densification

新帧通过 PnP 注册并做完 pose-only refinement 后，会对旧模型没有见过的区域补 anchor。

核心逻辑：

```text
上一帧渲染深度 pre_rendered_depth
    -> 反投影成 3D 点
    -> 投影到新帧
    -> 得到 occ_mask
```

这里的 `occ_mask=True` 表示：新帧这个像素能被上一帧已有几何覆盖到。

因此：

```python
densify_mask = occ_mask.view(-1) == 0
```

表示新帧中旧模型覆盖不到的区域，也就是新看到的地方。

随后：

```python
gaussians.densify_occlusion(viewpoint_cam, viewpoint_cam.depth_map, densify_mask)
```

会做：

```text
新帧新区域像素
    -> 使用新帧 depth_map 反投影成 3D 点
    -> 去掉深度边缘点
    -> 变换到世界坐标
    -> octree / voxel 采样
    -> 和已有 anchor 去重
    -> 初始化为新的 anchor
```

新 anchor 的初始参数：

```text
anchor: 新区域深度反投影得到的 3D 位置
scaling: 根据 voxel size 初始化
rotation: 单位四元数
opacity: 初值约为 0.1
anchor_feat: 全 0
offset: 全 0，形状为 n_offsets x 3
```

所以这一步可以理解为：  
**把新帧中新出现的 3D 点云转换成新的 Scaffold-GS anchor，加入 GaussianModel，后续再通过 local/global optimization 优化。**

### 6. 所有帧加入后的可选 pruning

while 结束后，如果：

```text
opt.pruning_ratio > 0
```

会统计每个 anchor 在所有训练视角中的 touched 次数，删除贡献最低的一部分。

默认：

```text
pruning_ratio = 0
```

所以默认不执行。

### 7. 最终 Refinement

最后进入完整序列精修：

```text
opt.iterations = 30000 + post_iter
first_iter = 30000
```

如果：

```text
post_iter = 20000
```

则：

```text
Refinement: iteration 30001 ~ 50000
```

开始前会执行：

```text
view.to_final()
```

把所有相机切到最终图像分辨率。

refinement 使用所有训练帧随机采样优化：

```text
训练视角: 所有 train cameras
loss: L1 + SSIM + scaling_reg + depth loss + 2D correspondence loss
优化: Gaussian 参数 + 相机 pose delta
```

refinement densification：

```text
update_until = 30000 + post_iter // 2 = 40000

统计: 30001 ~ 39999
adjust_anchor: 30100,30200,...,39900
清理统计缓存: 40000
40001 ~ 50000: 不再 densify，只做参数精修
```

同时每 400 步执行一次指数学习率衰减：

```text
30400, 30800, 31200, ...
```

### 总结

```text
init:
    用前 3 帧建立初始点云、位姿和 Gaussian。

while:
    一帧一帧加入新图像。
    每个新帧先 PnP 注册，再 pose refinement。
    然后 local 优化当前窗口，global 优化所有已注册帧。

refinement:
    所有帧加入后，切到最终分辨率，用完整序列继续精修。
```
