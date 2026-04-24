import logging
import sqlite3
import numpy as np
from tinytag import TinyTag
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("init_db")

MAP_PATH = "./data/processed/file_mapping.npy"
DB_PATH = "./data/music_meta.db"


def init_database():
    logger.warning(
        "此脚本操作旧的 data/music_meta.db，与当前主库 music_hash.db 无关。"
        "如需重建主库索引，请使用 batch_extract.py。"
    )

    if not os.path.exists(MAP_PATH):
        logger.error("找不到 file_mapping.npy，请先运行特征提取脚本。")
        return

    mapping = np.load(MAP_PATH)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS songs (
            faiss_id INTEGER PRIMARY KEY,
            file_path TEXT,
            title TEXT,
            artist TEXT
        )
    ''')
    cursor.execute('DELETE FROM songs')

    logger.info("开始构建旧版数据库，准备处理 %d 首歌曲...", len(mapping))

    for faiss_id, file_path in enumerate(mapping):
        try:
            tag = TinyTag.get(file_path)
            title = tag.title if tag.title else os.path.basename(file_path)
            artist = tag.artist if tag.artist else "未知歌手"
            cursor.execute(
                'INSERT INTO songs (faiss_id, file_path, title, artist) VALUES (?, ?, ?, ?)',
                (faiss_id, file_path, title, artist)
            )
            if faiss_id > 0 and faiss_id % 1000 == 0:
                logger.info("已存入: %d/%d...", faiss_id, len(mapping))
        except Exception:
            logger.exception("处理文件失败: %s", file_path)
            cursor.execute(
                'INSERT INTO songs (faiss_id, file_path, title, artist) VALUES (?, ?, ?, ?)',
                (faiss_id, file_path, os.path.basename(file_path), "读取失败")
            )

    conn.commit()
    conn.close()
    logger.info("旧版数据库构建完成: %s", DB_PATH)


if __name__ == "__main__":
    init_database()
