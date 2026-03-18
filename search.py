import faiss
import numpy as np
import torch
import librosa
import sqlite3
import os
from model import AudioHashNet

HASH_PATH = "./data/processed/all_music_hashes.npy"
DB_PATH = "./data/music_meta.db"
MODEL_WEIGHTS = "./checkpoints/hash_model_epoch_5.pth"


def load_system():
    print("⏳ 正在启动 AI 听歌识曲引擎...")
    fingerprints = np.load(HASH_PATH).astype('uint8')

    index = faiss.IndexBinaryFlat(128)
    index.add(fingerprints)

    # 🌟 核心升级：加载训练好的模型权重
    model = AudioHashNet()
    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=True))
    else:
        print("⚠️ 警告：未找到训练权重，系统仍在使用随机大脑！")
    model.eval()

    print("✅ 引擎启动完毕！准备好大显身手了。")
    return index, model


def search_music(test_audio_path, index, model, top_k=5):
    # 1. 还是读 15 秒
    y, sr = librosa.load(test_audio_path, sr=22050, duration=15.0)

    # 🌟 核心救星：音量归一化！无论录音多小声，强制把音量拉到最大标准值！
    y = librosa.util.normalize(y)

    # 2. 自动切除静音/底噪
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)

    # 3. 截取前 5 秒
    target_length = int(sr * 5.0)
    if len(y_trimmed) > target_length:
        y_final = y_trimmed[:target_length]
    else:
        y_final = np.pad(y_trimmed, (0, max(0, target_length - len(y_trimmed))))

    # 后面的提取保持不变
    spec = librosa.feature.melspectrogram(y=y_final, sr=sr, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    tensor = torch.FloatTensor(spec_db).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        hash_out = model(tensor)
        binary_code = (hash_out > 0).int().numpy()
        query_packed = np.packbits(binary_code, axis=1).astype('uint8')

    D, I = index.search(query_packed, k=top_k)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    results = []
    for i in range(len(I[0])):
        faiss_id = int(I[0][i])
        distance = int(D[0][i])

        if faiss_id != -1:
            cursor.execute("SELECT title, artist, file_path FROM songs WHERE faiss_id=?", (faiss_id,))
            row = cursor.fetchone()
            if row:
                results.append({
                    "rank": i + 1,
                    "title": row[0],
                    "artist": row[1],
                    "distance": distance,
                    "file_path": row[2]
                })
    conn.close()
    return results