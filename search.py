import os
import torch
import librosa
import numpy as np
import sqlite3
import faiss
from model import AudioHashNet

# --- ⚙️ 全局配置参数 ---
DB_PATH = "music_hash.db"
INDEX_PATH = "music_hash.index"
MODEL_PATH = "./checkpoints/hash_model_epoch_3.pth"  # ⚠️ 确保使用的是完美收敛的第3轮模型


def load_system():
    print("⏳ 正在启动检索引擎与加载 8 缸 AI 大脑...")
    try:
        index = faiss.read_index_binary(INDEX_PATH)
        print(f"✅ Faiss 索引库加载成功！当前库中包含 {index.ntotal} 条子指纹数据。")
    except Exception as e:
        print(f"❌ Faiss 索引加载失败: {e}")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioHashNet().to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("✅ 深度哈希模型权重加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return index, None

    return index, model


def search_music(test_audio_path, index, model, top_k=150):
    # ==========================================
    # 🌟 第一步：音频预处理 (素颜状态，不加特效，与建库绝对一致)
    # ==========================================
    y, sr = librosa.load(test_audio_path, sr=22050, duration=15.0)

    total_duration = len(y) / sr
    if total_duration < 5.0:
        y = np.pad(y, (0, int(sr * 5.0) - len(y)))
        total_duration = 5.0

    device = next(model.parameters()).device

    # ==========================================
    # 🌟 第二步：能量选秀！只取声音最饱满的前3个切片
    # ==========================================
    window_candidates = []
    for start_sec in range(0, max(1, int(total_duration - 4))):
        start_sample = int(start_sec * sr)
        end_sample = int((start_sec + 5.0) * sr)
        y_window = y[start_sample:end_sample]

        if len(y_window) < int(sr * 5.0):
            y_window = np.pad(y_window, (0, int(sr * 5.0) - len(y_window)))

        energy = np.mean(librosa.feature.rms(y=y_window))
        window_candidates.append((energy, y_window))

    window_candidates.sort(key=lambda x: x[0], reverse=True)
    best_windows = window_candidates[:3]

    queries = []
    for energy, y_window in best_windows:
        spec = librosa.feature.melspectrogram(y=y_window, sr=sr, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        tensor = torch.FloatTensor(spec_db).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            hash_out = model(tensor)

            # 🌟 核心破局魔法：必须和建库保持一致的中值过滤！
            threshold = hash_out.median()
            binary_code = (hash_out > threshold).int().cpu().numpy()

            query_packed = np.packbits(binary_code, axis=1).astype('uint8')
            queries.append(query_packed)

    if not queries:
        return []

    # ==========================================
    # 🌟 第三步：多重检索与投票合并
    # ==========================================
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    song_best_match = {}

    for query_packed in queries:
        D, I = index.search(query_packed, k=top_k)

        for i in range(len(I[0])):
            faiss_id = int(I[0][i])
            distance = int(D[0][i])

            if faiss_id != -1:
                cursor.execute("SELECT title, artist, file_path FROM songs WHERE faiss_id=?", (faiss_id,))
                row = cursor.fetchone()
                if row:
                    title, artist, file_path = row

                    # --------------------------------------------------------
                    # 🌟 核心修改区：生成前端能播放的网络 URL
                    # 1. 抹平系统差异：把 Windows 本地的反斜杠 '\' 全换成网址用的正斜杠 '/'
                    normalized_path = file_path.replace("\\", "/")

                    # 2. 截断与拼接：截取 'raw/' 后面的具体文件名，拼上咱们在 main.py 开通的 '/audio/' 专线
                    if "raw/" in normalized_path:
                        audio_url = "/audio/" + normalized_path.split("raw/")[-1]
                    else:
                        audio_url = ""  # 保底防崩机制
                    # --------------------------------------------------------

                    # 记录并去重，只保留同一首歌里汉明距离最小（最相似）的那一次撞击
                    if file_path not in song_best_match or distance < song_best_match[file_path]["distance"]:
                        song_best_match[file_path] = {
                            "title": title,
                            "artist": artist,
                            "distance": distance,
                            "file_path": file_path,
                            "audio_url": audio_url  # 👈 关键点：把播放链接装进字典返回给前端！
                        }
    conn.close()

    # ==========================================
    # 🌟 第四步：排序并输出最终 Top 5
    # ==========================================
    sorted_results = sorted(song_best_match.values(), key=lambda x: x["distance"])

    final_results = []
    for i, res in enumerate(sorted_results[:5]):
        res["rank"] = i + 1
        final_results.append(res)

    return final_results