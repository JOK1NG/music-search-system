"""
端到端业务层验证脚本
====================
从 data/raw 随机抽 N 首歌，整首上传到 /search 接口，统计 Top-1 是否
命中原曲。用于反坍塌训练完成后的最低门槛业务验证 —— 哈希指标再漂亮，
最终要让原曲自查百分百命中才能算闭环。

运行前提：FastAPI 服务已在 http://127.0.0.1:8000 启动
（即 python main.py）。

用法：
    python tools/e2e_search_test.py             # 默认抽 5 首
    python tools/e2e_search_test.py --n 20      # 自定义抽样数
"""
import argparse
import os
import random
import time

import requests

RAW_PATH = "./data/raw"
URL = "http://127.0.0.1:8000/search"
SEED = 42


def collect_audio_files():
    files = []
    for root, _, fs in os.walk(RAW_PATH):
        for f in fs:
            if f.lower().endswith((".mp3", ".wav")):
                files.append(os.path.join(root, f))
    return files


def hit_check(query_path, results, top_k=5):
    """判定 query 是否在 results 的前 top_k 命中。返回 (是否命中, Top-1 是否命中)"""
    fname = os.path.basename(query_path)
    title_guess = os.path.splitext(fname)[0].lower()

    top1_hit = False
    topk_hit = False
    for k, r in enumerate(results[:top_k]):
        title = (r.get("title") or "").lower()
        path = r.get("file_path") or ""
        match = (title_guess in title) or (title in title_guess and title) or \
                os.path.normcase(path).endswith(os.path.normcase(fname))
        if match:
            topk_hit = True
            if k == 0:
                top1_hit = True
            break
    return top1_hit, topk_hit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="抽样歌曲数量")
    args = parser.parse_args()

    random.seed(SEED)
    files = collect_audio_files()
    print(f"[OK] 数据库共 {len(files)} 首音频，随机抽 {args.n} 首作为 query")
    chosen = random.sample(files, min(args.n, len(files)))

    top1_hits = 0
    topk_hits = 0
    total_time = 0.0

    for i, path in enumerate(chosen, 1):
        fname = os.path.basename(path)
        print(f"\n[{i}/{args.n}] query = {fname}")

        with open(path, "rb") as f:
            t0 = time.time()
            try:
                resp = requests.post(URL, files={"file": (fname, f, "audio/mpeg")}, timeout=60)
            except Exception as e:
                print(f"   [ERR] HTTP 请求失败: {e}")
                continue
            dt = time.time() - t0

        total_time += dt
        if resp.status_code != 200:
            print(f"   [ERR] HTTP {resp.status_code}: {resp.text[:200]}")
            continue

        body = resp.json()
        results = body.get("data") or []
        if not results:
            print(f"   [MISS] 返回空结果, msg={body.get('message')}")
            continue

        top1 = results[0]
        print(f"   Top-1: {top1.get('title','')} ({dt*1000:.0f} ms)")

        is_top1, is_topk = hit_check(path, results, top_k=5)
        if is_top1:
            top1_hits += 1
            topk_hits += 1
            print(f"   ✅ Top-1 命中")
        elif is_topk:
            topk_hits += 1
            print(f"   ⚠️ 不是 Top-1 但在 Top-5 内")
            for k, r in enumerate(results[:5], 1):
                print(f"     {k}. {r.get('title','')}")
        else:
            print(f"   ❌ MISS。Top-5：")
            for k, r in enumerate(results[:5], 1):
                print(f"     {k}. {r.get('title','')}")

    print()
    print("=" * 60)
    print(f"Top-1 命中: {top1_hits}/{args.n} ({top1_hits/args.n*100:.1f}%)")
    print(f"Top-5 命中: {topk_hits}/{args.n} ({topk_hits/args.n*100:.1f}%)")
    print(f"平均响应:   {total_time/args.n*1000:.0f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
