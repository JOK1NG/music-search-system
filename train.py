"""
“完全体训”治治哈希空间坍塌：

  - in-batch hard negative mining（梳起难负例，让梯度不再交 easy 例交 0）
  - 量化损失 Lq = (|h| - 1)²     （驱 Tanh 输出靠近 ±1）
  - 位平衡损失 Lb = (mean_b h)²  （驱每位 0/1 均衡）
  - β·tanh(βz) 退火：β 从 1 线性升到 10，逼近 sign
  - CosineAnnealingLR + lr=1e-4 + epoch=20 + batch=128

训完后记得用 batch_extract.py 重建 music_hash.index（模型变了旧索引立刻失效）。
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import AudioTripletDataset
from model import AudioHashNet

# ----------------------------------------------------------
# 超参数区
# ----------------------------------------------------------
# v3: v1 (紧同曲) 与 v2 (撞码空间) 之间的折中路线
#   v1 实测: same=13.83 / uniq=88.7% / bit_dev=0.149 / cross=53.20 / 碰撞码=1886
#   v2 实测: same=16.92 / uniq=94.9% / bit_dev=0.079 / cross=61.65 / 碰撞码=706
#   v3 期望: same≤15  且  cross≥60  且  bit_dev≤0.10  且  碰撞码≤900
#   超参折中:
#     - MARGIN     v1=8.0 / v2=6.0 → v3=7.0   (和稀 triplet 间隙)
#     - LAMBDA_BAL v1=0.05 / v2=0.15 → v3=0.10  (位平衡取中量)
#     - BETA_END   v1=10 / v2=4 → v3=6           (退火不太过 也不不至)
#     - dataset.py 保持 v2 形态 (anchor 不增强, 因为这是正确方向)
NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
MARGIN = 7.0
LAMBDA_QUANT = 0.1   # 量化损失权重

LAMBDA_BAL = 0.10    # 位平衡损失权重 (v1=0.05 / v2=0.15 折中)
ALPHA_EASY = 0.3     # easy negative 损失权重，hard 拿剩下的 0.7
BETA_START = 1.0     # β·tanh 退火起点
BETA_END = 6.0       # 退火终点 (v1=10 / v2=4 折中)
NUM_WORKERS = 4


def compute_beta(epoch, total_epochs):
    """β 随 epoch 线性从 BETA_START 升到 BETA_END"""
    if total_epochs <= 1:
        return BETA_END
    frac = epoch / (total_epochs - 1)
    return BETA_START + (BETA_END - BETA_START) * frac


def hard_negative_mining(hash_a):
    """“完全体训”治治哈希空间坍塌：

  - in-batch hard negative mining（筛出难负例，让梯度不再被 easy 例主导为 0）
  - 量化损失 Lq = (|h| - 1)²     （驱 Tanh 输出靠近 ±1）
  - 位平衡损失 Lb = (mean_b h)²  （驱每位 0/1 均衡）
  - β·tanh(βz) 退火：β 从 1 线性升到 6，逼近 sign
  - CosineAnnealingLR + lr=1e-4 + epoch=10 + batch=128

训完后记得用 batch_extract.py 重建 music_hash.index（模型变了旧索引立刻失效）。
    in-batch hard negative：对每个 anchor i，在 batch 内除自己外 hash 距离最近的样本作为 hard neg。
    这些样本本身是别人的 anchor，属于不同歌。
    返回一个与 hash_a 同形状的 tensor，作为 hash_n。
    """
    with torch.no_grad():
        dist = torch.cdist(hash_a, hash_a, p=2)  # (B, B)
        dist.fill_diagonal_(float("inf"))
        idx = dist.argmin(dim=1)
    return hash_a[idx]


def batch_health_metrics(model, anchor, positive, eval_beta=1.0):
    """运行期快速估计哈希健康度：均位偏差、同曲距、bit-balance。

    为什么要临时切 eval() + β=1 重跑一次前向：
      - eval(): BatchNorm 不再用当前 batch 的波动统计，避免训练日志同曲距虚高
      - β=1:    与推理脚本 (search.py / select_best_epoch.py) 保持一致，
                不被 β 退火后期 tanh 饱和干扰
    v1 训练日志 same=20.8b 但实际诊断仅 13.25b，差异就出在 BN+β 评估模式不一致，
    本函数修复后，训练日志与离线诊断的数字场面对齐。
    """
    with torch.no_grad():
        was_training = model.training
        model.eval()
        hash_a = model(anchor, beta=eval_beta)
        hash_p = model(positive, beta=eval_beta)
        bin_a = (hash_a > hash_a.median(dim=1, keepdim=True).values).float()
        bin_p = (hash_p > hash_p.median(dim=1, keepdim=True).values).float()
        same_song_dist = (bin_a != bin_p).float().sum(dim=1).mean().item()
        bit_p1 = bin_a.mean(dim=0)  # 每位为 1 的比例
        bit_dev = (bit_p1 - 0.5).abs().mean().item()
        if was_training:
            model.train()
    return same_song_dist, bit_dev


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    dataset = AudioTripletDataset("./data/raw")
    use_gpu = device.type == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS if use_gpu else 0,
        pin_memory=use_gpu,
        drop_last=True,  # hard negative mining 需要 batch 不变
    )

    model = AudioHashNet().to(device)
    triplet = nn.TripletMarginLoss(margin=MARGIN, p=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print(f"🔥 开训 | epochs={NUM_EPOCHS} batch={BATCH_SIZE} lr={LEARNING_RATE} "
          f"margin={MARGIN} β={BETA_START}→{BETA_END}")
    print(f"   loss = triplet_easy*{ALPHA_EASY} + triplet_hard*{1-ALPHA_EASY:.1f} "
          f"+ Lq*{LAMBDA_QUANT} + Lb*{LAMBDA_BAL}")

    os.makedirs("./checkpoints", exist_ok=True)
    for epoch in range(NUM_EPOCHS):
        beta = compute_beta(epoch, NUM_EPOCHS)
        model.train()
        t0 = time.time()
        agg = {"total": 0.0, "easy": 0.0, "hard": 0.0, "q": 0.0, "b": 0.0,
               "same": 0.0, "dev": 0.0, "n": 0}

        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)

            optimizer.zero_grad()

            hash_a = model(anchor, beta=beta)
            hash_p = model(positive, beta=beta)
            hash_n_easy = model(negative, beta=beta)
            hash_n_hard = hard_negative_mining(hash_a)  # 从当前 hash_a 重采样

            loss_easy = triplet(hash_a, hash_p, hash_n_easy)
            loss_hard = triplet(hash_a, hash_p, hash_n_hard)

            # 量化损失：表达式在 ±1 附近为 0
            loss_q = ((hash_a.abs() - 1).pow(2).mean()
                      + (hash_p.abs() - 1).pow(2).mean()
                      + (hash_n_easy.abs() - 1).pow(2).mean()) / 3.0

            # 位平衡损失：每位在 batch 内均值趋近 0（代表 0/1 各占一半）
            loss_b = (hash_a.mean(dim=0).pow(2).mean()
                      + hash_p.mean(dim=0).pow(2).mean()
                      + hash_n_easy.mean(dim=0).pow(2).mean()) / 3.0

            loss = (ALPHA_EASY * loss_easy
                    + (1 - ALPHA_EASY) * loss_hard
                    + LAMBDA_QUANT * loss_q
                    + LAMBDA_BAL * loss_b)
            loss.backward()
            optimizer.step()

            same, dev = batch_health_metrics(model, anchor, positive, eval_beta=1.0)
            agg["total"] += loss.item()
            agg["easy"] += loss_easy.item()
            agg["hard"] += loss_hard.item()
            agg["q"] += loss_q.item()
            agg["b"] += loss_b.item()
            agg["same"] += same
            agg["dev"] += dev
            agg["n"] += 1

            if (batch_idx + 1) % 5 == 0:
                print(f"  Ep[{epoch+1}/{NUM_EPOCHS}] B[{batch_idx+1}/{len(dataloader)}] "
                      f"β={beta:.2f} lr={scheduler.get_last_lr()[0]:.2e} "
                      f"L={loss.item():.3f} (E={loss_easy.item():.2f} H={loss_hard.item():.2f} "
                      f"Q={loss_q.item():.3f} B={loss_b.item():.3f}) "
                      f"same={same:.1f}b dev={dev:.3f}")

        scheduler.step()
        n = max(agg["n"], 1)
        dt = time.time() - t0
        print(f"💾 Epoch {epoch+1} 完成 ({dt/60:.1f} min) | "
              f"avg_loss={agg['total']/n:.3f} "
              f"easy={agg['easy']/n:.2f} hard={agg['hard']/n:.2f} "
              f"Lq={agg['q']/n:.3f} Lb={agg['b']/n:.3f} | "
              f"同曲距={agg['same']/n:.1f}bits  bit-dev={agg['dev']/n:.3f}")

        # v3 专属前缀保存，不覆盖 v2 的 hash_model_epoch_*.pth
        torch.save(model.state_dict(), f"./checkpoints/hash_model_v3_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    train()