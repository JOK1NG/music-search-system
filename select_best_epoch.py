"""
快诊脚本：在不重建 Faiss 索引的前提下，对每个 epoch 的 .pth 快速评估哈希质量，
挑出 "同曲距最小 × 跨歌距最大 × bit-balance 最均" 的最优 epoch。

预计总耗时 ~3 分钟（RTX 4080 + 30s 波形缓存 + 200 首×5 片段采样）。
"""
import argparse
import os
import glob
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import _utf8_console  # noqa: F401  强制 stdout/stderr 走 UTF-8，规避 Windows cp936 重定向丢字符（详见模块内注释）

import numpy as np
import torch
import librosa

from model import AudioHashNet

CACHE_PATH = "./data/waveform_cache_30s"
CKPT_DIR = "./checkpoints"
SR = 22050
WINDOW_DUR = 5.0
WINDOW_SAMPLES = int(SR * WINDOW_DUR)
N_SONGS = 200          # 采样几首歌
N_PER_SONG = 5         # 每首歌抽几个 5s 片段
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_full_waveform(path, full_samples=int(SR * 30.0)):
    y = np.load(path)
    if len(y) < full_samples:
        y = np.pad(y, (0, full_samples - len(y)))
    return y[:full_samples]


def random_window(y_full, rng):
    max_start = max(0, len(y_full) - WINDOW_SAMPLES)
    start = rng.randint(0, max_start) if max_start > 0 else 0
    return y_full[start:start + WINDOW_SAMPLES]


def extract_mel(y):
    spec = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return torch.FloatTensor(spec_db).unsqueeze(0).unsqueeze(0)  # (1,1,128,T)


def build_sample_pool():
    """
    一次性采样 N_SONGS 首歌×N_PER_SONG 个片段的 Mel（CPU），后面 20 个模型全部复用。
    返回两个 tensor:
      - mel_pool: (N_SONGS*N_PER_SONG, 1, 128, T) 频谱
      - song_id : (N_SONGS*N_PER_SONG,)         所属歌曲索引
    """
    print(f"⏳ 采样 {N_SONGS} 首歌×{N_PER_SONG} 片段的 Mel···")
    rng = random.Random(SEED)
    cache_files = sorted(glob.glob(os.path.join(CACHE_PATH, "*.npy")))
    if len(cache_files) < N_SONGS:
        print(f"⚠️ 缓存中只有 {len(cache_files)} 首，不够 {N_SONGS}，取全部")
    chosen = rng.sample(cache_files, min(N_SONGS, len(cache_files)))

    mels, song_ids = [], []
    for sid, path in enumerate(chosen):
        y_full = load_full_waveform(path)
        for _ in range(N_PER_SONG):
            y_win = random_window(y_full, rng)
            mels.append(extract_mel(y_win))
            song_ids.append(sid)
    mel_pool = torch.cat(mels, dim=0)            # (N_SONGS*N_PER_SONG, 1, 128, T)
    song_id = torch.tensor(song_ids, dtype=torch.long)
    print(f"[OK] 采样完成    mel_pool={tuple(mel_pool.shape)}  song_id={tuple(song_id.shape)}")
    return mel_pool, song_id


@torch.no_grad()
def run_model(model, mel_pool, batch=128):
    """分批推理，返回连续 Tanh 输出 (N, 128) 和二值化编码 (N, 128) bool"""
    model.eval()
    outs = []
    for i in range(0, len(mel_pool), batch):
        x = mel_pool[i:i + batch].to(device, non_blocking=True)
        h = model(x, beta=1.0)  # 推理用 β=1，避免 hash 推理时的非线性激活
        outs.append(h.cpu())
    h_cont = torch.cat(outs, dim=0)              # (N, 128)
    median = h_cont.median(dim=1, keepdim=True).values
    h_bin = (h_cont > median)                    # (N, 128) bool
    return h_cont, h_bin


def evaluate_codes(h_bin, song_id):
    """计算 4 项指标：同曲平均 / 跨歌平均 / bit-balance / 跨歌 distance=0 比例"""
    h_bin = h_bin.numpy()  # (N, 128) bool
    n = len(h_bin)

    same_pairs, cross_pairs = [], []
    cross_zero = 0
    rng = random.Random(SEED + 1)
    n_pairs = 5000

    for _ in range(n_pairs):
        i = rng.randrange(n)
        # 同曲不同片段
        same_idx = [k for k in range(n) if song_id[k] == song_id[i] and k != i]
        if same_idx:
            j = rng.choice(same_idx)
            same_pairs.append(int((h_bin[i] != h_bin[j]).sum()))
        # 跨歌
        diff_idx = rng.randrange(n)
        while song_id[diff_idx] == song_id[i]:
            diff_idx = rng.randrange(n)
        d = int((h_bin[i] != h_bin[diff_idx]).sum())
        cross_pairs.append(d)
        if d == 0:
            cross_zero += 1

    p1 = h_bin.mean(axis=0)             # 每位为 1 的比例
    bit_dev = float(np.abs(p1 - 0.5).mean())

    return {
        "same_mean": float(np.mean(same_pairs)),
        "same_med": float(np.median(same_pairs)),
        "cross_mean": float(np.mean(cross_pairs)),
        "cross_med": float(np.median(cross_pairs)),
        "bit_dev": bit_dev,
        "cross_zero_pct": cross_zero / len(cross_pairs) * 100,
        "uniq_pct": len(set(map(tuple, h_bin.tolist()))) / len(h_bin) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="挑 checkpoint 最佳 epoch")
    parser.add_argument("--prefix", default="hash_model_epoch",
                        help="checkpoint 文件名前缀（不含 _epoch_X.pth）。"
                             "v1/v2 默认 hash_model_epoch，v3 试 hash_model_v3_epoch")
    args = parser.parse_args()

    print(f"[GPU] device = {device}")
    mel_pool, song_id = build_sample_pool()

    # 列出所有 .pth，按 epoch 数升序
    pattern = os.path.join(CKPT_DIR, f"{args.prefix}_*.pth")
    ckpts = sorted(glob.glob(pattern),
                   key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))
    if not ckpts:
        print(f"[ERR] 未找到区配 {pattern} 的 .pth, 请检查 ./checkpoints/ 目录")
        return
    print(f"[INFO] 匹配到 {len(ckpts)} 个 checkpoint，前缀 = {args.prefix}")

    model = AudioHashNet().to(device)

    print()
    header = (f"{'epoch':>5} | {'same_mean':>9} {'same_med':>8} | "
              f"{'cross_mean':>10} {'cross_med':>9} | {'bit_dev':>7} | "
              f"{'cross=0%':>8} | {'uniq%':>6}")
    print(header)
    print("-" * len(header))

    rows = []
    for ckpt in ckpts:
        epoch = int(ckpt.rsplit("_", 1)[-1].split(".")[0])
        model.load_state_dict(torch.load(ckpt, map_location=device))
        _, h_bin = run_model(model, mel_pool)
        m = evaluate_codes(h_bin, song_id.numpy())
        rows.append((epoch, m))
        print(f"{epoch:>5} | {m['same_mean']:>9.2f} {m['same_med']:>8.0f} | "
              f"{m['cross_mean']:>10.2f} {m['cross_med']:>9.0f} | {m['bit_dev']:>7.3f} | "
              f"{m['cross_zero_pct']:>7.2f}% | {m['uniq_pct']:>5.1f}%")

    # 综合评分：同曲最小 × 跨歌最大 × bit-balance 最均 × uniq 最高
    def score(m):
        return (-m["same_mean"] / 64.0          # +1: same=0
                - abs(m["cross_mean"] - 64) / 64.0  # +1: cross=64
                - m["bit_dev"] * 2              # bit_dev 越小越好
                + m["uniq_pct"] / 100.0         # +1: uniq 100%
                - m["cross_zero_pct"] / 100.0)  # 跨歌 distance=0 越少越好

    best_epoch, best_m = max(rows, key=lambda r: score(r[1]))
    print()
    print(f"[BEST] 推荐 epoch = {best_epoch}, same_mean={best_m['same_mean']:.2f}  "
          f"cross_mean={best_m['cross_mean']:.2f}  bit_dev={best_m['bit_dev']:.3f}  "
          f"uniq={best_m['uniq_pct']:.1f}%")
    print(f"   下一步修改 search.py / batch_extract.py 的 MODEL_PATH = "
          f'"./checkpoints/{args.prefix}_{best_epoch}.pth"')


if __name__ == "__main__":
    main()
