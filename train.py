import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AudioTripletDataset
from model import AudioHashNet

from config import (
    TRAIN_LR, TRAIN_MARGIN, TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE, TRAIN_NUM_WORKERS,
)

logger = logging.getLogger("train")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    dataset = AudioTripletDataset()
    use_gpu = device.type == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=TRAIN_NUM_WORKERS if use_gpu else 0,
        pin_memory=use_gpu
    )

    model = AudioHashNet().to(device)
    criterion = nn.TripletMarginLoss(margin=TRAIN_MARGIN, p=2)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR)

    logger.info("开始训练！学习率: %s, margin: %s, epochs: %d", TRAIN_LR, TRAIN_MARGIN, TRAIN_EPOCHS)

    for epoch in range(TRAIN_EPOCHS):
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
                    "Epoch [%d/%d], Batch [%d/%d], Loss: %.4f",
                    epoch + 1, TRAIN_EPOCHS, batch_idx + 1, len(dataloader), loss.item()
                )

        avg_loss = total_loss / len(dataloader)
        logger.info("Epoch %d 完成！平均 Loss: %.4f", epoch + 1, avg_loss)
        torch.save(model.state_dict(), f"./checkpoints/hash_model_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
