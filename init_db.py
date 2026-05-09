import os

import _utf8_console  # noqa: F401  强制 stdout/stderr 走 UTF-8，规避 Windows cp936 重定向丢字符（详见模块内注释）

import sqlite3
import numpy as np
from tinytag import TinyTag

# 1. 路径配置
MAP_PATH = "./data/processed/file_mapping.npy"
DB_PATH = "./data/music_meta.db"  # 生成的 SQLite 数据库文件


def init_database():
    # 加载映射表 (Faiss 返回的 ID 就是这个数组的索引)
    if not os.path.exists(MAP_PATH):
        print("❌ 错误：找不到 file_mapping.npy，请先运行特征提取脚本。")
        return

    mapping = np.load(MAP_PATH)

    # 2. 连接 SQLite 数据库 (如果没有会自动创建)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 3. 创建数据表 (包含 Faiss_ID, 路径, 歌名, 歌手)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS songs (
            faiss_id INTEGER PRIMARY KEY,
            file_path TEXT,
            title TEXT,
            artist TEXT
        )
    ''')

    # 清空旧数据（防止你重复运行这个脚本导致数据叠加）
    cursor.execute('DELETE FROM songs')

    print(f"📦 开始构建数据库，准备处理 {len(mapping)} 首歌曲...")

    # 4. 遍历映射表，读取 mp3 标签并存入数据库
    for faiss_id, file_path in enumerate(mapping):
        try:
            # 读取音频自带的元数据
            tag = TinyTag.get(file_path)

            # 如果 mp3 里没有自带歌名，就用文件名凑合一下
            title = tag.title if tag.title else os.path.basename(file_path)
            artist = tag.artist if tag.artist else "未知歌手"

            # 插入数据库
            cursor.execute('''
                INSERT INTO songs (faiss_id, file_path, title, artist)
                VALUES (?, ?, ?, ?)
            ''', (faiss_id, file_path, title, artist))

            if faiss_id > 0 and faiss_id % 1000 == 0:
                print(f"⌛ 已存入: {faiss_id}/{len(mapping)}...")

        except Exception as e:
            # 万一某个文件损坏，也要存个保底数据，确保 Faiss_ID 严格对齐
            cursor.execute('''
                INSERT INTO songs (faiss_id, file_path, title, artist)
                VALUES (?, ?, ?, ?)
            ''', (faiss_id, file_path, os.path.basename(file_path), "读取失败"))

    # 5. 提交保存并关闭连接
    conn.commit()
    conn.close()
    print("\n" + "=" * 30)
    print("✅ 数据库构建完美竣工！")
    print(f"💾 数据库文件已生成: {DB_PATH}")
    print("=" * 30)


if __name__ == "__main__":
    init_database()