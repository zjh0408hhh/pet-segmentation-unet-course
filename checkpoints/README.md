# 模型权重

本仓库未提供预训练权重，可按根目录 README 训练生成。

`baseline_best.pth` 和 `improved_best.pth` 用于评估。`*_resume.pth` 保存模型、优化器、调度器与随机状态，用于续训。

本次实验最佳权重的 SHA256、模型结构及轮次见 [manifest.json](manifest.json)。取得相应权重后，可用 `--checkpoint-dir` 指定评估目录。
