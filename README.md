# AI 音乐相似度检索系统

本项目为轻量级音乐音频检索服务，包含 FastAPI 后端、Faiss 检索引擎与基于 CNN 的音频哈希模型。

## 训练参数说明（`config.py`）

训练相关参数可通过环境变量覆盖：

- `TRAIN_LR`：学习率，默认 `1e-5`
- `TRAIN_MARGIN`：TripletLoss margin，默认 `5.0`
- `TRAIN_EPOCHS`：最大训练轮数，默认 `3`
- `TRAIN_BATCH_SIZE`：批大小，默认 `64`
- `TRAIN_NUM_WORKERS`：`DataLoader` worker 数量，默认 `4`
- `TRAIN_NOISE_FACTOR`：随机噪声增强系数，默认 `0.01`
- `TRAIN_DROPOUT`：模型 `Dropout` 比例，默认 `0.3`
- `TRAIN_PATIENCE`：早停耐心值，默认 `5`
- `TRAIN_MIN_DELTA`：早停最小改进阈值，默认 `1e-4`
- `TRAIN_CHECKPOINT_DIR`：checkpoint 保存目录，默认 `./checkpoints`
- `TRAIN_RESUME_CHECKPOINT`：恢复点文件名，默认 `latest.ckpt`
- `TRAIN_LOG_DIR`：TensorBoard 日志目录，默认 `./runs`

示例（Linux / macOS）：

```bash
export TRAIN_DROPOUT=0.3
export TRAIN_PATIENCE=6
export TRAIN_MIN_DELTA=0.0005
python train.py
```

```bash
tensorboard --logdir "$TRAIN_LOG_DIR"
```

说明：`TRAIN_RESUME_CHECKPOINT` 默认会从 `TRAIN_CHECKPOINT_DIR` 下读取并恢复模型与优化器状态（若不存在则从头训练）。
