# 数据与结果图的来源、改动和许可

数据：Oxford-IIIT Pet，Omkar M. Parkhi、Andrea Vedaldi、Andrew Zisserman、C. V. Jawahar，Cats and Dogs，CVPR 2012。

来源：https://www.robots.ox.ac.uk/~vgg/data/pets/

数据官网公布的许可证：Creative Commons Attribution-ShareAlike 4.0 International（CC BY-SA 4.0）。许可链接：https://creativecommons.org/licenses/by-sa/4.0/ 。原始图片版权仍归原权利人所有。仓库不分发原始数据包或原始照片。

`comparison/test_predictions.png`、`comparison/最大改善案例.png`、`comparison/最大退步案例.png` 为基于该数据的派生可视化：标注经过 128×128 最近邻缩放，原 trimap 的宠物与边界合并为前景；另展示本项目两组模型预测的二值掩码。公开版本已省略原始照片列。对这些派生图的贡献按 CC BY-SA 4.0 提供，不暗示原作者认可本项目。

训练曲线由本项目运行记录生成，数值见对应的 `history.json`。源代码采用根目录 GPL v3 许可证，数据及派生图采用上述 CC BY-SA 4.0 许可。
