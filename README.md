# U-Net 宠物图像分割：复现与卷积轻量化

基于 [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) 的固定版本，保留原始 U-Net 作为基线，新增深层和解码端的深度可分离卷积版本。在 Oxford-IIIT Pet 上用相同配置分别训练 10 轮，以验证集选择模型，最后统一评估独立测试集。

本项目将原项目的 Carvana 汽车分割任务适配为宠物前景分割。代码来源与许可证见 [SOURCES.md](SOURCES.md)，修改说明见 [CHANGES.md](CHANGES.md)。

## 实验结果

| 独立测试集指标 | 原版 | 改进版 |
|---|---:|---:|
| 前景 IoU | 71.59% | 77.48% |
| 前景 Dice | 83.44% | 87.31% |
| 参数量 | 31,037,698 | 6,216,898 |
| CPU 单图预测中位耗时 | 131.85 ms | 49.92 ms |

测试集共 3,669 张，两组均使用验证 Dice 最好的第 9 轮权重。IoU / Dice 由整个测试集累计的前景 TP、FP、FN 计算。速度测试使用 Ryzen 5 5500U、6 线程、FP32、128×128、batch=1；对同一张预加载图片预热后交替计时 30 次，取中位数，不含读图和保存。

本次实验使用单个随机种子，每种结构训练一次，结果仅代表该配置。逐图 IoU 有 2,709 张改善、953 张退步、7 张持平。

- [完整测试记录](results/comparison/test_comparison.json)、[逐图指标](results/comparison/per_image_test.csv)
- [整体对比图](results/comparison/test_predictions.png)
- [最大改善案例](results/comparison/最大改善案例.png)、[最大退步案例](results/comparison/最大退步案例.png)

## 目录

```text
models/unet/          原版网络（保持上游文件不变）
models/efficient_unet.py  改进版：14 个普通卷积分解为深度可分离卷积
upstream_utils/      原版 Dice 实现
pet_data.py          RGB、128×128 和二分类标注处理
experiment.py        共享损失、验证指标及绘图
train.py             原版与改进版共用的训练入口
evaluate.py          固定最佳模型后的统一测试和速度评估
visualize_cases.py   对比图和改善/退步案例
prepare_data.py      压缩包校验、安全解压和固定数据划分
splits/              实验实际使用的训练/验证/测试清单
results/             实验记录、指标与对比图
checkpoints/         权重说明及 SHA256 清单
runs/                新运行产生的输出（不提交）
utils/runtime.py     训练进程管理
```

仓库包含源码与实验结果，数据集需从官方页面下载。本仓库未提供预训练权重，复现实验需先运行训练脚本。

## 环境安装

实验环境：Windows 11、Python 3.12.14、PyTorch 2.5.1+cpu、16GB 内存。安装 Python 3.12 后，在项目根目录使用 PowerShell 执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`requirements.txt` 列出直接依赖，`requirements-lock.txt` 固定实验环境的全部包版本，均使用官方 PyTorch CPU 下载源。项目已在 Windows 验证；Linux 可用 `python3.12 -m venv .venv` 创建环境，并使用 `.venv/bin/python`。

下列示例为简洁起见写作 `python`，请替换为所建环境的 Python，例如 `.\.venv\Scripts\python.exe`。

## 数据准备

从 [Oxford-IIIT Pet 官方页面](https://www.robots.ox.ac.uk/~vgg/data/pets/) 下载 `images.tar.gz` 和 `annotations.tar.gz`，合计约 811 MB，使用数据集自带的分割标注。

```powershell
python prepare_data.py --archives "D:\datasets\pet-archives" --data-root "D:\datasets\oxford-iiit-pet"
```

路径可替换为自己的位置。脚本校验 MD5、解压并检查图片与掩码。压缩包有 7,390 张 JPG，本实验使用官方清单中的 7,349 张：从官方 trainval 按品种、seed=2026 划分训练集 2,944 张、验证集 736 张，保留官方测试集 3,669 张。具体名单见 `splits/`。

标注原值 1（宠物）和 3（边界）统一为前景 1，原值 2 为背景 0。图片统一 RGB，双三次缩放到 128×128 并除以 255；掩码使用最近邻缩放。两组使用同一规则。

## 训练与续训

```powershell
python train.py --model upstream_unet --data-root "D:\datasets\oxford-iiit-pet" --epochs 10
python train.py --model efficient_unet --data-root "D:\datasets\oxford-iiit-pet" --epochs 10
```

两组均从随机初始化训练，使用 seed=2026、batch=2、CPU 6 线程、FP32、交叉熵 + 原版 Dice、RMSprop（lr=1e-5、momentum=0.999、weight_decay=1e-8）、梯度裁剪 1.0、ReduceLROnPlateau（验证 Dice，patience=5，每轮调用一次），不使用数据增强。

训练记录写入 `runs/baseline/` 和 `runs/improved/`，权重保存到 `checkpoints/`。继续已有训练时添加 `--resume`：

```powershell
python train.py --model efficient_unet --data-root "D:\datasets\oxford-iiit-pet" --epochs 10 --resume
```

程序每轮保存完整训练状态，支持从最近保存的轮次恢复；中途未完成的轮次需重新运行。

训练前可先检查配置和数据读取：

```powershell
python train.py --model efficient_unet --data-root "D:\datasets\oxford-iiit-pet" --check-only
```

## 评估与可视化

两组训练结束后，固定各自验证 Dice 最好的模型，执行：

```powershell
python evaluate.py --data-root "D:\datasets\oxford-iiit-pet"
python visualize_cases.py --data-root "D:\datasets\oxford-iiit-pet"
```

默认读取两组的 `runs/*/summary.json` 和 `checkpoints/*_best.pth`，输出到 `runs/comparison/`；已有评估结果时拒绝覆盖。模型选择仅使用验证集。对比图展示固定间隔选取的 4 个测试样本，以及 IoU 改善最大、退步最大的各 1 个案例。

若已取得本次实验的权重，可使用归档训练记录复核结果：

```powershell
python evaluate.py --data-root "D:\datasets\oxford-iiit-pet" --experiments-dir results --checkpoint-dir "D:\models\petseg" --output-dir runs/recheck
python visualize_cases.py --data-root "D:\datasets\oxford-iiit-pet" --checkpoint-dir "D:\models\petseg" --comparison-dir runs/recheck
```

`--checkpoint-dir` 指定权重目录，本次实验权重的 SHA256 见 [checkpoints/manifest.json](checkpoints/manifest.json)。

## 完整性检查

[MANIFEST.json](MANIFEST.json) 记录文件 SHA256。运行 `python verify_release.py` 可检查文件完整性、上游源码、数据划分和归档指标。发布前整理及验证范围见 [CHANGES.md](CHANGES.md)。

## 许可

代码采用 [GPL v3](LICENSE)，数据及派生图采用 CC BY-SA 4.0。仓库对比图展示标注与预测掩码；本地运行可生成含原图的对比图。图像来源、处理方式与许可见 [图像许可说明](results/ASSETS_LICENSE.md)。
