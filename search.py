import logging
import sqlite3

import torch
import numpy as np
import faiss

from model import AudioHashNet
from config import DB_PATH, INDEX_PATH, MODEL_PATH, TOP_K, TOP_N_RESULTS, SAMPLE_RATE
from preprocessing import (
    load_audio, pad_audio,
    build_mel_spectrogram, spec_to_tensor, binarize_hash,
    slice_best_windows, extract_hash_from_full_spectrum,
)

logger = logging.getLogger("search")


def load_system():
    logger.info("正在启动检索引擎与加载 AI 大脑...")
    try:
        index = faiss.read_index_binary(INDEX_PATH)
        logger.info("Faiss 索引库加载成功！当前库中包含 %d 条子指纹数据。", index.ntotal)
    except Exception as e:
        logger.exception("Faiss 索引加载失败")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioHashNet().to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        logger.info("深度哈希模型权重加载成功！")
    except Exception as e:
        logger.exception("模型加载失败")
        return index, None

    return index, model


def _collect_db_songs(conn, all_faiss_ids):
    """批量 SQL: 一次 IN 查询替代 N 次单条查询"""
    if not all_faiss_ids:
        return {}
    placeholders = ",".join("?" for _ in all_faiss_ids)
    cursor = conn.execute(
        f"SELECT faiss_id, title, artist, file_path FROM songs WHERE faiss_id IN ({placeholders})",
        all_faiss_ids
    )
    return {row[0]: row for row in cursor.fetchall()}


def search_music(test_audio_path, index, model, top_k=TOP_K):
    if index is None or model is None:
        logger.error("检索引擎未就绪")
        return []

    device = next(model.parameters()).device

    # --- 第一步: 加载完整音频 ---
    y, sr = load_audio(test_audio_path, duration=15.0)

    total_duration = len(y) / sr
    if total_duration < 5.0:
        y = pad_audio(y, 5.0)
        total_duration = 5.0

    # --- 第二步: 一次计算完整大频谱 (P2-16) ---
    full_spec = build_mel_spectrogram(y, sr)

    # --- 第三步: 按能量选取最佳窗口，从大频谱切片提取哈希 ---
    best_windows = slice_best_windows(y)

    queries = []
    for _energy, _y_window, start_sec in best_windows:
        query_packed = extract_hash_from_full_spectrum(full_spec, model, device, start_sec)
        queries.append(query_packed)

    if not queries:
        return []

    # --- 第四步: 多重检索与投票合并 (P2-14: 批量 SQL) ---
    conn = sqlite3.connect(DB_PATH)
    song_best_match = {}

    all_faiss_ids = set()
    query_results = []
    for query_packed in queries:
        D, I = index.search(query_packed, k=top_k)
        for i in range(len(I[0])):
            faiss_id = int(I[0][i])
            distance = int(D[0][i])
            if faiss_id != -1:
                all_faiss_ids.add(faiss_id)
                query_results.append((faiss_id, distance))

    if all_faiss_ids:
        db_songs = _collect_db_songs(conn, list(all_faiss_ids))
        for faiss_id, distance in query_results:
            row = db_songs.get(faiss_id)
            if row:
                title, artist, file_path = row[1], row[2], row[3]
                normalized_path = file_path.replace("\\", "/")
                audio_url = "/audio/" + normalized_path.split("raw/")[-1] if "raw/" in normalized_path else ""
                if file_path not in song_best_match or distance < song_best_match[file_path]["distance"]:
                    song_best_match[file_path] = {
                        "title": title,
                        "artist": artist,
                        "distance": distance,
                        "file_path": file_path,
                        "audio_url": audio_url,
                    }

    conn.close()

    # --- 第五步: 排序输出 Top N ---
    sorted_results = sorted(song_best_match.values(), key=lambda x: x["distance"])[:TOP_N_RESULTS]
    for i, res in enumerate(sorted_results):
        res["rank"] = i + 1

    return sorted_results
