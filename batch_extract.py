import os
import torch
import librosa
import numpy as np
import sqlite3
import faiss
from model import AudioHashNet
from tinytag import TinyTag

# --- ⚙️ 配置路径 ---
RAW_PATH = "./data/raw"
DB_PATH = "music_hash.db"
INDEX_PATH = "music_hash.index"
MODEL_PATH = "./checkpoints/hash_model_epoch_3.pth"  # ⚠️ 确保使用的是完美收敛的第3轮模型


def extract_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 加载强力 AI 大脑中... 使用设备: {device}")

    model = AudioHashNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 1. 重建数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS songs")
    cursor.execute('''
        CREATE TABLE songs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            faiss_id INTEGER,
            title TEXT,
            artist TEXT,
            file_path TEXT,
            offset REAL
        )
    ''')

    # 2. 重新初始化 Faiss 极速索引 (128位 = 16 bytes)
    index = faiss.IndexBinaryFlat(16 * 8)

    # 3. 收集文件
    music_files = []
    for root, dirs, files in os.walk(RAW_PATH):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav')):
                music_files.append(os.path.join(root, file))

    print(f"🎵 共找到 {len(music_files)} 首歌曲，准备执行「滑动窗口」指纹提取...")
    faiss_id_counter = 0

    # 4. 遍历提取并存库
    with torch.no_grad():
        for i, file_path in enumerate(music_files):
            try:
                # 提取原生标签
                try:
                    tag = TinyTag.get(file_path)
                    title = tag.title if tag.title else os.path.basename(file_path)
                    artist = tag.artist if tag.artist else "Unknown Artist"
                except Exception:
                    title = os.path.basename(file_path)
                    artist = "Unknown"

                # 提取前 30 秒进行建库切片
                y, sr = librosa.load(file_path, sr=22050, duration=30.0)
                total_duration = len(y) / sr

                if total_duration < 5.0:
                    y = np.pad(y, (0, int(sr * 5.0) - len(y)))
                    total_duration = 5.0

                # 滑动窗口切香肠
                for start_sec in range(0, int(total_duration - 4)):
                    start_sample = int(start_sec * sr)
                    end_sample = int((start_sec + 5.0) * sr)
                    y_window = y[start_sample:end_sample]

                    spec = librosa.feature.melspectrogram(y=y_window, sr=sr, n_mels=128)
                    spec_db = librosa.power_to_db(spec, ref=np.max)
                    tensor = torch.FloatTensor(spec_db).unsqueeze(0).unsqueeze(0).to(device)

                    hash_out = model(tensor)

                    # 🌟 核心破局魔法：动态中值哈希 (克服特征漂移！)
                    threshold = hash_out.median()
                    binary_code = (hash_out > threshold).int().cpu().numpy()

                    query_packed = np.packbits(binary_code, axis=1).astype('uint8')

                    # 存入 Faiss 和 SQLite
                    index.add(query_packed)
                    cursor.execute(
                        "INSERT INTO songs (faiss_id, title, artist, file_path, offset) VALUES (?, ?, ?, ?, ?)",
                        (faiss_id_counter, title, artist, file_path, start_sec)
                    )
                    faiss_id_counter += 1

                if (i + 1) % 50 == 0:
                    print(f"⏳ 进度: [{i + 1}/{len(music_files)}] 已生成 {faiss_id_counter} 条子指纹...")

            except Exception as e:
                pass  # 忽略坏文件

    # 保存革命果实
    faiss.write_index_binary(index, INDEX_PATH)
    conn.commit()
    conn.close()
    print(f"🎉 史诗级指纹库建库完成！8000首歌曲裂变成了 {faiss_id_counter} 条抗错位子指纹！")


if __name__ == "__main__":
    extract_features()