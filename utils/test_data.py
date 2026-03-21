import os
import librosa
import numpy as np

# 配置路径
RAW_DATA_PATH = "../data/raw"


def scan_music_files(root_path):
    music_list = []
    # os.walk 可以自动遍历所有子文件夹 [cite: 535]
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.mp3'):
                full_path = os.path.join(root, file)
                music_list.append(full_path)
    return music_list


if __name__ == "__main__":
    songs = scan_music_files(RAW_DATA_PATH)
    print(f"✅ 扫描完毕！共发现 {len(songs)} 首歌曲。")

    if len(songs) > 0:
        print(f"📂 第一首歌的路径是: {songs[0]}")
        # 验证一下之前学到的采样率：22050Hz [cite: 187, 517]
        y, sr = librosa.load(songs[0], sr=22050, duration=1.0)
        print(f"🎵 采样率验证正常: {sr}Hz")