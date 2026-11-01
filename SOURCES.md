# 代码来源与复现范围

上游：[milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)，固定提交 `21d7850f2af30a9695bbeea75f3136aa538cfc4a`。

以下文件保留上游内容，未修改：`models/unet/__init__.py`、`models/unet/unet_model.py`、`models/unet/unet_parts.py`、`upstream_utils/dice_score.py`。文件摘要见 `results/source_audit.json` 与 `MANIFEST.json`。上游 GNU GPL v3 许可证完整保存在 `LICENSE`，本派生代码以 GPL v3 分发。

原项目默认使用 Carvana 汽车数据。本项目基于其模型及 Dice 实现，适配 Oxford-IIIT Pet 数据、CPU 训练和统一评估。

新增内容包括数据处理与固定划分、训练/验证入口、轻量化网络、统一评估和案例可视化。轻量化采用已有的深度可分离卷积方法，具体实现见 [CHANGES.md](CHANGES.md)。

数据来源：[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)，官网公布的许可证为 CC BY-SA 4.0，原图版权仍归原权利人。数据与派生图的许可独立于代码许可，详见 [图像许可说明](results/ASSETS_LICENSE.md)。

数据引用：Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, C. V. Jawahar. *Cats and Dogs*. CVPR, 2012。

官方归档 MD5 的核查来源：[Torchvision Oxford-IIIT Pet 实现](https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/oxford_iiit_pet.html)。二分类中将边界合并入前景的约定也见于 [segmentation_models.pytorch 的 Oxford Pet 读取代码](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/datasets/oxford_pet.py)。
