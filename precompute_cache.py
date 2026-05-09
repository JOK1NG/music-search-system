"""
预计算音频波形缓存脚本
=======================
将 data/raw/ 下所有 MP3/WAV 解码为标准化的 NumPy 波形数组，
存入 data/waveform_cache/ 目录，跳过训练时最昂贵的 librosa.load() MP3 解码步骤。

用法：python precompute_cache.py
"""

import os

import _utf8_console  # noqa: F401  强制 stdout/stderr 走 UTF-8，规避 Windows cp936 重定向丢字符（详见模块内注释）

import numpy as np
import librosa

RAW_PATH = "./data/raw"
CACHE_PATH = "./data/waveform_cache_30s"  # 🌟 与升级后的 dataset.py 对齐（随机切窗 30s 训练）
SR = 22050
DURATION = 30.0  # 训练需要 30s 全段，才能在中间随机抽 5s 起点


def precompute():
    os.makedirs(CACHE_PATH, exist_ok=True)
    target_length = int(SR * DURATION)

    music_files = []
    for root, dirs, files in os.walk(RAW_PATH):
        for f in files:
            if f.lower().endswith(('.mp3', '.wav')):
                music_files.append(os.path.join(root, f))

    print(f"🎵 共找到 {len(music_files)} 个音频文件，开始预计算波形缓存...")

    cached, skipped, failed = 0, 0, 0
    for i, file_path in enumerate(music_files):
        # 用文件的相对路径生成缓存文件名（保持唯一性，避免同名文件冲突）
        rel_path = os.path.relpath(file_path, RAW_PATH)
        cache_name = rel_path.replace(os.sep, "__") + ".npy"
        cache_file = os.path.join(CACHE_PATH, cache_name)

        if os.path.exists(cache_file):
            skipped += 1
            continue

        try:
            y, _ = librosa.load(file_path, sr=SR, duration=DURATION)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            y = y[:target_length]

            np.save(cache_file, y)
            cached += 1
        except Exception as e:
            failed += 1

        if (i + 1) % 200 == 0:
            print(f"⏳ 进度: [{i + 1}/{len(music_files)}] 已缓存 {cached} / 跳过 {skipped} / 失败 {failed}")

    print(f"\n🎉 波形缓存完成！")
    print(f"   ✅ 新缓存: {cached}")
    print(f"   ⏭️ 已存在跳过: {skipped}")
    print(f"   ❌ 失败: {failed}")
    print(f"   📂 缓存目录: {CACHE_PATH}")


if __name__ == "__main__":
    precompute()
