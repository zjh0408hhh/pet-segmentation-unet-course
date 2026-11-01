# 相对原项目的代码修改

## 核心模型修改

`models/efficient_unet.py` 新增 `DepthwiseSeparableConv` 和 `EfficientUNet`。普通 3×3 卷积一次完成空间与通道混合；改进块先用 groups=in_channels 的 3×3 逐通道卷积，再用 1×1 逐点卷积混合通道。忽略偏置时，参数量由 `9 × Cin × Cout` 变为 `9 × Cin + Cin × Cout`。

保留 inc、down1 的原始普通卷积，替换 down2、down3、down4 与 up1、up2、up3、up4 的 14 个普通卷积。通道宽度、跳跃连接、BN/ReLU、转置上采样和输出类别保持一致。保留浅层卷积的设计动机是保留早期纹理和颜色特征提取能力；并未通过额外消融单独证明这一设计的贡献。

参数由 31,037,698 减少到 6,216,898。`models/factory.py` 统一选择原版和改进版模型，实验结果见 README 与 `results/`。

## 实验适配与公平对照

`pet_data.py` 适配 Oxford trimap、RGB、固定输入尺寸；`prepare_data.py` 验证官方压缩包和独立划分。`experiment.py` 共用交叉熵 + 上游 Dice 和累计前景指标。`train.py` 使用同一套优化器、随机种子、数据、轮数和调度器，只有模型选择不同。原项目的数据加载、联网日志与每轮多次验证调整为 CPU 适用的加载、离线记录与每轮一次验证；两组统一采用这些调整。

最终评估使用各组验证 Dice 最好的模型。速度测试使用同一张预加载图片、相同精度与线程配置，预热后交替计时。逐图指标记录在 `results/comparison/per_image_test.csv`。

## 代码组织与验证

训练入口统一为 `train.py`，损失、评估与绘图集中在 `experiment.py`。新运行输出到 `runs/`，本次实验记录保存在 `results/`。

归档结果来自代码整理前完成的训练。整理后核对了两组检查点的配置与恢复兼容性，并验证真实图片的前向计算；未重新进行完整训练。
