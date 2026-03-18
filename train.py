import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import AudioHashNet
from dataset import MusicTripletDataset
import os


# ==========================================
# 1. 定义复合哈希损失函数 (Hash Loss) [cite: 198-206]
# ==========================================
class HashLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, margin=2.0):
        super(HashLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        # 相似度严师：Triplet Loss，让 Anchor 靠近 Positive，远离 Negative [cite: 208-209]
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, h_anchor, h_pos, h_neg):
        # 1. Pairwise 相似度损失
        l_pair = self.triplet_loss(h_anchor, h_pos, h_neg)

        # 2. Quantization 量化损失：逼迫浮点数输出靠近 1 或 -1 [cite: 203-204]
        # 公式推导：(|h| - 1)^2
        l_quant = torch.mean((torch.abs(h_anchor) - 1.0) ** 2) + \
                  torch.mean((torch.abs(h_pos) - 1.0) ** 2) + \
                  torch.mean((torch.abs(h_neg) - 1.0) ** 2)

        # 3. Balance 平衡损失：逼迫模型输出的 1 和 -1 数量大概相等（均值趋近于0） [cite: 205-206]
        l_bal = torch.mean(h_anchor.mean(dim=1) ** 2) + \
                torch.mean(h_pos.mean(dim=1) ** 2) + \
                torch.mean(h_neg.mean(dim=1) ** 2)

        return l_pair + self.alpha * l_quant + self.beta * l_bal


# ==========================================
# 2. 核心训练循环
# ==========================================
def train():
    # 基础配置
    RAW_PATH = "./data/raw"
    BATCH_SIZE = 16  # 每次看16组歌
    EPOCHS = 5  # 训练轮数
    LEARNING_RATE = 1e-5  # 初始学习率 [cite: 210](修改前为1e-4)

    # 初始化设备 (如果有 NVIDIA 显卡会自动用 GPU，否则用 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 训练设备已就绪: {device}")

    # 加载数据集和模型
    print("📦 正在准备三元组数据集 (可能会稍等几秒读取文件列表)...")
    dataset = MusicTripletDataset(RAW_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = AudioHashNet().to(device)
    criterion = HashLoss(alpha=0.1, beta=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)


    print("🚀 正式开始训练！")

    # 开始跑 Epoch
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (spec_a, spec_p, spec_n) in enumerate(dataloader):
            # 将数据推入设备
            spec_a, spec_p, spec_n = spec_a.to(device), spec_p.to(device), spec_n.to(device)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播：大脑分别听三段音频
            h_a = model(spec_a)
            h_p = model(spec_p)
            h_n = model(spec_n)

            # 计算复合损失
            loss = criterion(h_a, h_p, h_n)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 每处理 5 个 Batch 打印一次日志
            if (batch_idx + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # 保存每个 Epoch 的模型权重
        save_dir = "./checkpoints"
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        torch.save(model.state_dict(), f"{save_dir}/hash_model_epoch_{epoch + 1}.pth")
        print(f"💾 Epoch {epoch + 1} 完成！模型已保存，平均 Loss: {running_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    train()