import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import AudioTripletDataset
from model import AudioHashNet
from config import (
    TRAIN_LR, TRAIN_MARGIN, TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE, TRAIN_NUM_WORKERS,
    TRAIN_DROPOUT, TRAIN_PATIENCE, TRAIN_MIN_DELTA,
    TRAIN_CHECKPOINT_DIR, TRAIN_RESUME_CHECKPOINT, TRAIN_LOG_DIR,
)

logger = logging.getLogger("train")


def save_checkpoint(model, optimizer, epoch, best_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
    }, path)
    logger.info("检查点已保存: %s", path)


def load_checkpoint(path, model, optimizer, device):
    if not os.path.exists(path):
        logger.info("未找到检查点，从头开始训练")
        return 0, float("inf")

    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("best_loss", checkpoint.get("loss", float("inf")))
        logger.info("已从检查点恢复: epoch=%d, best_loss=%.6f", start_epoch, best_loss)
        return start_epoch, best_loss
    except Exception:
        logger.exception("检查点加载失败，将从头开始训练")
        return 0, float("inf")


def train():
    os.makedirs(TRAIN_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TRAIN_LOG_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    dataset = AudioTripletDataset()
    use_gpu = device.type == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=TRAIN_NUM_WORKERS if use_gpu else 0,
        pin_memory=use_gpu,
    )

    model = AudioHashNet(dropout_rate=TRAIN_DROPOUT).to(device)
    criterion = nn.TripletMarginLoss(margin=TRAIN_MARGIN, p=2)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR)

    resume_path = os.path.join(TRAIN_CHECKPOINT_DIR, TRAIN_RESUME_CHECKPOINT)
    start_epoch, best_loss = load_checkpoint(resume_path, model, optimizer, device)

    writer = SummaryWriter(log_dir=TRAIN_LOG_DIR)
    patience_counter = 0

    logger.info(
        "开始训练！lr=%s, margin=%s, epochs=%d, patience=%d",
        TRAIN_LR, TRAIN_MARGIN, TRAIN_EPOCHS, TRAIN_PATIENCE,
    )

    for epoch in range(start_epoch, TRAIN_EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            hash_a = model(anchor)
            hash_p = model(positive)
            hash_n = model(negative)

            loss = criterion(hash_a, hash_p, hash_n)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                logger.debug(
                    "Epoch [%d/%d] Batch [%d/%d] Loss: %.4f",
                    epoch + 1, TRAIN_EPOCHS, batch_idx + 1, len(dataloader), loss.item(),
                )

        avg_loss = total_loss / len(dataloader)
        logger.info("Epoch %d 完成 | Avg Loss: %.6f | Best: %.6f", epoch + 1, avg_loss, best_loss)

        writer.add_scalar("Loss/train", avg_loss, epoch + 1)
        writer.add_scalar("Loss/best", best_loss, epoch + 1)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch + 1)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch + 1)

        torch.save(model.state_dict(), os.path.join(TRAIN_CHECKPOINT_DIR, f"hash_model_epoch_{epoch + 1}.pth"))

        # 早停逻辑
        if avg_loss < best_loss - TRAIN_MIN_DELTA:
            best_loss = avg_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch + 1, best_loss, resume_path)
            logger.info("🎯 新最佳模型！loss=%.6f", best_loss)
        else:
            patience_counter += 1
            logger.info("⏳ 无改善 (%d/%d)", patience_counter, TRAIN_PATIENCE)
            if patience_counter >= TRAIN_PATIENCE:
                logger.info("早停触发！在第 %d 个 epoch 停止训练", epoch + 1)
                break

    writer.close()
    logger.info("训练完成！最佳 Loss: %.6f", best_loss)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
