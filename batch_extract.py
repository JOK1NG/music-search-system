import logging
import os

import torch
import numpy as np
import sqlite3
import faiss
from tinytag import TinyTag

from model import AudioHashNet
from config import DB_PATH, INDEX_PATH, MODEL_PATH, RAW_PATH, N_MELS, SAMPLE_RATE
from preprocessing import load_audio, pad_audio, build_mel_spectrogram, spec_to_tensor, binarize_hash

logger = logging.getLogger("batch_extract")


def extract_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("加载 AI 大脑中... 使用设备: %s", device)

    model = AudioHashNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # --- 重建数据库 ---
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
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_songs_faiss_id ON songs(faiss_id)
    ''')

    index = faiss.IndexBinaryFlat(16 * 8)

    # --- 收集文件 ---
    music_files = []
    for root, dirs, files in os.walk(RAW_PATH):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav')):
                music_files.append(os.path.join(root, file))

    logger.info("共找到 %d 首歌曲，准备执行滑动窗口指纹提取...", len(music_files))
    faiss_id_counter = 0
    failed_files = []

    with torch.no_grad():
        for i, file_path in enumerate(music_files):
            try:
                try:
                    tag = TinyTag.get(file_path)
                    title = tag.title if tag.title else os.path.basename(file_path)
                    artist = tag.artist if tag.artist else "Unknown Artist"
                except Exception:
                    title = os.path.basename(file_path)
                    artist = "Unknown"

                # --- 加载前 30 秒 ---
                y, sr = load_audio(file_path, duration=30.0)
                total_duration = len(y) / sr

                if total_duration < 5.0:
                    y = pad_audio(y, 5.0)
                    total_duration = 5.0

                # --- P2-15: 批量推理 —— 堆叠所有窗口一次推入模型 ---
                windows_tensors = []
                window_offsets = []
                for start_sec in range(0, int(total_duration - 4)):
                    start_sample = int(start_sec * sr)
                    end_sample = int((start_sec + 5.0) * sr)
                    y_window = y[start_sample:end_sample]

                    if len(y_window) < int(sr * 5.0):
                        y_window = np.pad(y_window, (0, int(sr * 5.0) - len(y_window)))

                    spec_db = build_mel_spectrogram(y_window, sr)
                    tensor = spec_to_tensor(spec_db)
                    windows_tensors.append(tensor)
                    window_offsets.append(start_sec)

                if not windows_tensors:
                    continue

                batch = torch.cat(windows_tensors, dim=0).to(device)
                hash_outs = model(batch)
                binary_codes = (hash_outs > hash_outs.median(dim=1, keepdim=True)[0]).int().cpu().numpy()
                packed = np.packbits(binary_codes, axis=1).astype('uint8')

                # --- 批量写入 Faiss 和 SQLite ---
                for j, query_packed in enumerate(packed):
                    index.add(np.expand_dims(query_packed, axis=0))
                    cursor.execute(
                        "INSERT INTO songs (faiss_id, title, artist, file_path, offset) VALUES (?, ?, ?, ?, ?)",
                        (faiss_id_counter, title, artist, file_path, window_offsets[j])
                    )
                    faiss_id_counter += 1

                if (i + 1) % 50 == 0:
                    logger.info("进度: [%d/%d] 已生成 %d 条子指纹...", i + 1, len(music_files), faiss_id_counter)

            except Exception:
                logger.exception("处理文件失败: %s", file_path)
                failed_files.append(file_path)

    # --- 保存 ---
    faiss.write_index_binary(index, INDEX_PATH)
    conn.commit()
    conn.close()

    if failed_files:
        logger.warning("以下 %d 个文件处理失败:\n%s", len(failed_files), "\n".join(failed_files))
    logger.info("指纹库建库完成！%d 首歌 → %d 条子指纹", len(music_files), faiss_id_counter)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    extract_features()
