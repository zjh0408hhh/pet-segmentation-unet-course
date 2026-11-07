# 宠物图像分割：U-Net 复现与轻量化

图像处理课程项目。将上游 U-Net 从汽车掩码预测适配到宠物前景分割，输入一张 RGB 图像，输出宠物与背景的二值掩码。两种模型用同一套数据划分和训练流程，比较分割质量、参数量和 CPU 推理速度。

## 数据集

[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) 包含 37 个品种。本实验使用官方清单中的 7,349 张图像。从官方训练验证集按品种划分出训练集 2,944 张、验证集 736 张，保留官方测试集 3,669 张。[划分清单](部署文档/数据划分/)

输入图像缩放至 128×128。原始标注中的宠物区域和边界合并为前景，其余为背景，缩放掩码时使用最近邻插值。[数据处理代码](源代码/pet_data.py)

## 代码改动

| 部分 | 实现 |
| --- | --- |
| 基线 | 保留[上游 U-Net 网络结构](源代码/models/unet/)，作为对照。 |
| 轻量版 | 将编码器后 3 个下采样模块和解码器 4 个上采样模块中的 14 个普通 3×3 卷积，分别替换为逐通道 3×3 卷积和 1×1 卷积。保留浅层卷积、各层通道数和跳跃连接。[模型实现](源代码/models/efficient_unet.py) |
| 训练与评估 | 两种模型共用[训练入口](源代码/train.py)和[评估入口](源代码/evaluate.py)。以验证集 Dice 选出各自的最佳权重，再统一评估独立测试集。 |

## 实验结果

两种模型均从随机初始化训练 10 轮，使用相同随机种子（2026）、批大小（2）和损失函数（交叉熵与 Dice 损失之和）。两组验证 Dice 最佳的权重均来自第 9 轮。表中 IoU 和 Dice 由整个测试集累计的前景像素统计计算。

| 模型 | 前景 IoU | 前景 Dice | 参数量 | 单图 CPU 推理中位耗时 |
| --- | ---: | ---: | ---: | ---: |
| 标准 U-Net | 71.59% | 83.44% | 31,037,698 | 131.85 ms |
| 轻量 U-Net | 77.48% | 87.31% | 6,216,898 | 49.92 ms |

轻量版参数量减少 **79.97%**，单图前向推理提速 **2.64 倍**，前景 IoU 提高 **5.89 个百分点**。逐图比较中，2,709 张的 IoU 提高，953 张下降，7 张持平。每种结构仅训练一次，表中数值对应本次实验设置。

速度测试使用 Ryzen 5 5500U、CPU 6 线程、FP32，对同一张预加载的 128×128 图像分别预热 5 次，再交替计时，每个模型测量 30 次。耗时仅包含模型前向计算。

- [测试指标](结果/测试对比/测试指标.json) · [预测计时](结果/测试对比/预测计时.json) · [逐图指标](结果/测试对比/逐图指标.csv)
- [预测对比](结果/测试对比/预测对比.png) · [最大改善案例](结果/测试对比/最大改善案例.png) · [最大退步案例](结果/测试对比/最大退步案例.png)
- [原版模型训练记录](结果/原版模型/) · [改进模型训练记录](结果/改进模型/)

## 复现实验

实验环境为 Windows 11、Python 3.12 和 PyTorch 2.5.1+cpu。在项目根目录打开 PowerShell。从[数据集官网](https://www.robots.ox.ac.uk/~vgg/data/pets/)下载 `images.tar.gz` 和 `annotations.tar.gz`，放入 `数据压缩包` 目录。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r 部署文档/requirements-lock.txt
.\.venv\Scripts\python.exe 源代码/prepare_data.py --archives 数据压缩包 --data-root 数据集
.\.venv\Scripts\python.exe 源代码/train.py --model upstream_unet --data-root 数据集 --epochs 10
.\.venv\Scripts\python.exe 源代码/train.py --model efficient_unet --data-root 数据集 --epochs 10
.\.venv\Scripts\python.exe 源代码/evaluate.py --data-root 数据集
.\.venv\Scripts\python.exe 源代码/visualize_cases.py --data-root 数据集
```

训练会生成评估所需的权重，新实验输出保存到 `结果/新实验/`。续训和复核已有权重的方法见[运行方法](部署文档/运行方法.txt)。

## 来源与许可

代码基于 [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet/tree/21d7850f2af30a9695bbeea75f3136aa538cfc4a)，按 [GNU GPL v3](LICENSE) 分发。数据集出处和结果图的许可见[许可与来源](部署文档/许可与来源.txt)。
