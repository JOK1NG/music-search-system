import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AudioTripletDataset
from model import AudioHashNet


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # 1. 加载数据集
    dataset = AudioTripletDataset("./data/raw")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # 2. 初始化 8 缸发动机网络
    model = AudioHashNet().to(device)

    # 🌟 核心修正 1：健康的及格线 margin=5.0
    criterion = nn.TripletMarginLoss(margin=5.0, p=2)

    # 🌟 核心修正 2：极其关键的“微步学习率” lr=1e-5 ！！！感谢你的火眼金睛！
    learning_rate = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 🌟 核心修正 3：只需 3 轮康复训练
    num_epochs = 3

    print(f"🔥 开始康复训练！学习率: {learning_rate}, 及格线: 5.0")

    # 3. 开始炼丹
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()

            # 提取三者的哈希特征
            hash_a = model(anchor)
            hash_p = model(positive)
            hash_n = model(negative)

            # 计算三元组损失
            loss = criterion(hash_a, hash_p, hash_n)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"💾 Epoch {epoch + 1} 完成！平均 Loss: {avg_loss:.4f}")

        # 保存这轮的权重
        torch.save(model.state_dict(), f"./checkpoints/hash_model_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    train()