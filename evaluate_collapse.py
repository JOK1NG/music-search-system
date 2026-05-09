"""
哈希空间坍塌诊断脚本
======================

用现有的 music_hash.index 和 music_hash.db 评估当前 128 位哈希码的健康度。
输出 5 项核心指标：

  [A] bit-balance        每一位 0/1 分布偏差（理想 0.5）
  [B] uniqueness         唯一哈希码 / 总条数（理想 100%）
  [C] collision-multi    碰撞码的多重度分布
  [D] same-song   汉明距离 同一首歌不同窗口（应 << 64，越小越好）
  [E] cross-song  汉明距离 不同歌曲不同窗口（应 ≈ 64，越接近越健康）

完全只读，不修改任何文件。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import _utf8_console  # noqa: F401  强制 stdout/stderr 走 UTF-8，规避 Windows cp936 重定向丢字符（详见模块内注释）

import sqlite3
import random
from collections import Counter, defaultdict

import numpy as np
import faiss


INDEX_PATH = "music_hash.index"
DB_PATH = "music_hash.db"
HASH_BITS = 128
SEED = 42
N_SAME_SONG_PAIRS = 500     # [D] 抽样
N_CROSS_SONG_PAIRS = 5000   # [E] 抽样


def load_codes():
    index = faiss.read_index_binary(INDEX_PATH)
    codes = faiss.vector_to_array(index.xb).reshape(index.ntotal, index.code_size)
    return index, codes  # codes: (N, 16) uint8


def hamming(a, b):
    """逐对汉明距离；a,b: (k, 16) uint8 → (k,) int"""
    xor = np.bitwise_xor(a, b)
    return np.unpackbits(xor, axis=1).sum(axis=1)


def metric_bit_balance(codes):
    bits = np.unpackbits(codes, axis=1)        # (N, 128)
    p1 = bits.mean(axis=0)                     # 每位是 1 的概率
    deviation = np.abs(p1 - 0.5)
    print(f"[A] bit-balance  mean|p-0.5| = {deviation.mean():.4f}  "
          f"max|p-0.5| = {deviation.max():.4f}  "
          f"(<=0.05 健康, >0.15 报警)")
    worst5 = np.argsort(-deviation)[:5]
    print(f"    最偏的 5 位: idx={worst5.tolist()}, p1={p1[worst5].round(3).tolist()}")


def metric_uniqueness(codes):
    n = len(codes)
    n_unique = len(np.unique(codes, axis=0))
    ratio = n_unique / n
    print(f"[B] uniqueness   {n_unique}/{n} = {ratio*100:.2f}%  "
          f"(>=99% 健康, <=80% 严重坍塌)")
    return n_unique


def metric_collision_multi(codes, faiss_id_to_song):
    """统计完全相同的哈希码涉及多少首不同歌"""
    code_to_songs = defaultdict(set)
    for fid, code in enumerate(codes):
        song = faiss_id_to_song.get(fid)
        if song is None:
            continue
        code_to_songs[code.tobytes()].add(song)
    multi = [len(s) for s in code_to_songs.values() if len(s) >= 2]
    if not multi:
        print("[C] collision    无跨歌碰撞，干净")
        return
    cnt = Counter(multi)
    print(f"[C] collision    存在跨歌哈希碰撞的码数 = {len(multi)}（涉及 ≥2 首不同歌）")
    print(f"    碰撞多重度分布 (k=多少首歌共享同一码): "
          f"{dict(sorted(cnt.items())[:10])}{' …' if len(cnt) > 10 else ''}")
    print(f"    最严重的一个码同时出现在 {max(multi)} 首不同歌里")


def metric_same_song(codes, song_to_faiss_ids, n_pairs=500):
    songs = [s for s, ids in song_to_faiss_ids.items() if len(ids) >= 2]
    if not songs:
        print("[D] same-song    无可用样本")
        return
    rng = random.Random(SEED)
    a, b = [], []
    for _ in range(n_pairs):
        s = rng.choice(songs)
        i, j = rng.sample(song_to_faiss_ids[s], 2)
        a.append(codes[i]); b.append(codes[j])
    d = hamming(np.array(a), np.array(b))
    print(f"[D] same-song    n={n_pairs}  mean={d.mean():.2f} bits  "
          f"median={int(np.median(d))}  std={d.std():.2f}  "
          f"min={int(d.min())}  max={int(d.max())}  "
          f"(理想 <8, >32 说明同曲不同片段已经塌得稀烂)")


def metric_cross_song(codes, song_to_faiss_ids, n_pairs=5000):
    songs = list(song_to_faiss_ids.keys())
    if len(songs) < 2:
        print("[E] cross-song   歌曲不足")
        return
    rng = random.Random(SEED + 1)
    a, b = [], []
    zeros = 0
    for _ in range(n_pairs):
        s1, s2 = rng.sample(songs, 2)
        i = rng.choice(song_to_faiss_ids[s1])
        j = rng.choice(song_to_faiss_ids[s2])
        a.append(codes[i]); b.append(codes[j])
    d = hamming(np.array(a), np.array(b))
    zeros = int((d == 0).sum())
    print(f"[E] cross-song   n={n_pairs}  mean={d.mean():.2f} bits  "
          f"median={int(np.median(d))}  std={d.std():.2f}  "
          f"min={int(d.min())}  max={int(d.max())}  "
          f"distance=0 命中 {zeros} 对（{zeros/n_pairs*100:.2f}%）  "
          f"(理想 mean≈64, distance=0 应≈0)")


def main():
    print("⏳ 读取 faiss 索引中…")
    index, codes = load_codes()
    print(f"✅ ntotal={index.ntotal}, code_size={index.code_size} bytes "
          f"({index.code_size*8} bits)\n")

    print("⏳ 读取 songs 表中…")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT faiss_id, file_path FROM songs")
    rows = cur.fetchall()
    conn.close()
    faiss_id_to_song = {fid: fp for fid, fp in rows}
    song_to_faiss_ids = defaultdict(list)
    for fid, fp in rows:
        song_to_faiss_ids[fp].append(fid)
    print(f"✅ 共 {len(faiss_id_to_song)} 条 faiss_id，"
          f"{len(song_to_faiss_ids)} 首歌，"
          f"平均每首 {len(faiss_id_to_song)/len(song_to_faiss_ids):.1f} 个窗口\n")

    print("=" * 60)
    print("🔬 哈希健康度诊断")
    print("=" * 60)
    metric_bit_balance(codes)
    print()
    metric_uniqueness(codes)
    print()
    metric_collision_multi(codes, faiss_id_to_song)
    print()
    metric_same_song(codes, song_to_faiss_ids, N_SAME_SONG_PAIRS)
    print()
    metric_cross_song(codes, song_to_faiss_ids, N_CROSS_SONG_PAIRS)
    print()
    print("=" * 60)
    print("📌 诊断完成。后续任何模型/loss 改动都可重跑此脚本，对比 [A]~[E] 的变化。")


if __name__ == "__main__":
    main()
