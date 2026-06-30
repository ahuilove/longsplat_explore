### Convert to 3DGS Format
转换成标准3dgs的结果质量和模型大小都会有变化
LongSplat uses an anchor + MLP structure for efficient reconstruction. We provide a conversion script to transform LongSplat results into the standard 3DGS format, which outputs a `point_cloud.ply` file that can be used with general 3DGS viewers.

**Note**: Converting to 3DGS format will change both quality and model size. We recommend applying pre-pruning to reduce the model size before conversion.

```bash
# Convert LongSplat output to 3DGS format
python convert_3dgs.py -m PATH_TO_TRAINED_MODEL --prune_ratio 0.6
```

### 计算指标的方式
- RPE_trans 代码里乘了 100
- RPE_rot 从弧度转成了角度
- ATE 是对齐后的绝对轨迹误差

总结一句：LongSplat 的测试集图像质量是在估计出来的 test pose 上渲染后和 GT 图像比；test pose 本身是用训练好的 Gaussian + MASt3R 匹配 + PnP + photometric
refinement 得到的；pose 指标则用 COLMAP sparse 里的位姿作为 GT，经过尺度/轨迹对齐后计算 ATE/RPE。

