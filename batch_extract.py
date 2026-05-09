import os
import logging

import _utf8_console  # noqa: F401  强制 stdout/stderr 走 UTF-8，规避 Windows cp936 重定向丢字符（详见模块内注释）

import torch
import sqlite3
import faiss
from model import AudioHashNet
from tinytag import TinyTag
from config import DB_PATH, INDEX_PATH, MODEL_PATH, RAW_PATH, WINDOW_DURATION
import preprocessing

BUILD_TABLE = "_songs_build"

logger = logging.getLogger(__name__)


CREATE_BUILD_TABLE_SQL = f'''
    CREATE TABLE {BUILD_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        faiss_id INTEGER,
        title TEXT,
        artist TEXT,
        file_path TEXT,
        offset REAL
    )
'''


def _reset_build_table(cursor):
    cursor.execute(f"DROP TABLE IF EXISTS {BUILD_TABLE}")
    cursor.execute(CREATE_BUILD_TABLE_SQL)


def _cleanup_build_artifacts(conn, temp_index_path):
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {BUILD_TABLE}")
    conn.commit()
    if os.path.exists(temp_index_path):
        os.remove(temp_index_path)


def _commit_build(conn, temp_index_path):
    cursor = conn.cursor()
    backup_index_path = INDEX_PATH + ".bak"
    moved_existing_index = False
    cursor.execute("BEGIN")
    try:
        cursor.execute("DROP TABLE IF EXISTS songs")
        cursor.execute(f"ALTER TABLE {BUILD_TABLE} RENAME TO songs")

        if os.path.exists(backup_index_path):
            os.remove(backup_index_path)
        if os.path.exists(INDEX_PATH):
            os.replace(INDEX_PATH, backup_index_path)
            moved_existing_index = True
        os.replace(temp_index_path, INDEX_PATH)

        conn.commit()
    except Exception:
        conn.rollback()
        if moved_existing_index and os.path.exists(backup_index_path):
            if os.path.exists(INDEX_PATH):
                os.remove(INDEX_PATH)
            os.replace(backup_index_path, INDEX_PATH)
        raise
    finally:
        if os.path.exists(backup_index_path):
            os.remove(backup_index_path)


def extract_features():
    temp_index_path = INDEX_PATH + ".tmp"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 加载强力 AI 大脑中... 使用设备: {device}")

    model = AudioHashNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    if os.path.exists(temp_index_path):
        os.remove(temp_index_path)

    # 1. 先写临时构建表，成功前不动正式 songs 表
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    _reset_build_table(cursor)
    conn.commit()

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
    failed_files = []

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
                y, sr = preprocessing.load_audio(file_path, duration=30.0)
                total_duration = len(y) / sr

                if total_duration < WINDOW_DURATION:
                    y = preprocessing.pad_audio(y, WINDOW_DURATION, sr=sr)
                    total_duration = WINDOW_DURATION

                # 滑动窗口切香肠
                for start_sec in range(0, max(1, int(total_duration - WINDOW_DURATION + 1))):
                    start_sample = int(start_sec * sr)
                    end_sample = int((start_sec + WINDOW_DURATION) * sr)
                    y_window = y[start_sample:end_sample]

                    y_window = preprocessing.pad_audio(y_window, WINDOW_DURATION, sr=sr)
                    query_packed = preprocessing.extract_hash_from_window(y_window, model, device, sr=sr)

                    # 存入 Faiss 和 SQLite
                    index.add(query_packed)
                    cursor.execute(
                        f"INSERT INTO {BUILD_TABLE} (faiss_id, title, artist, file_path, offset) VALUES (?, ?, ?, ?, ?)",
                        (faiss_id_counter, title, artist, file_path, start_sec)
                    )
                    faiss_id_counter += 1

                if (i + 1) % 50 == 0:
                    print(f"⏳ 进度: [{i + 1}/{len(music_files)}] 已生成 {faiss_id_counter} 条子指纹...")

            except Exception as e:
                failed_files.append((file_path, repr(e)))
                logger.exception("指纹提取失败，文件已记录，本轮不会覆盖正式索引: %s", file_path)
                print(f"❌ 文件处理失败: {file_path} | {e!r}")

    if failed_files:
        print(f"🛑 建库中止：{len(failed_files)} 个文件失败，正式 songs 表和索引保持不变。")
        for file_path, error in failed_files:
            print(f"   - {file_path}: {error}")
        _cleanup_build_artifacts(conn, temp_index_path)
        conn.close()
        return False

    if faiss_id_counter == 0:
        print("🛑 建库中止：没有生成任何子指纹，正式 songs 表和索引保持不变。")
        _cleanup_build_artifacts(conn, temp_index_path)
        conn.close()
        return False

    # 保存临时索引；只有写入成功后才切换正式 songs 表和正式索引
    try:
        faiss.write_index_binary(index, temp_index_path)
        conn.commit()
        _commit_build(conn, temp_index_path)
    except Exception:
        logger.exception("建库提交失败，正式 songs 表和索引保持不变")
        print("🛑 建库提交失败，正式 songs 表和索引保持不变。")
        _cleanup_build_artifacts(conn, temp_index_path)
        conn.close()
        return False

    conn.close()
    print(f"🎉 史诗级指纹库建库完成！8000首歌曲裂变成了 {faiss_id_counter} 条抗错位子指纹！")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if extract_features() else 1)
