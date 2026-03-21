import sqlite3
import os
from tinytag import TinyTag

DB_PATH = "../music_hash.db"


def fix_metadata():
    print("🛠️ 开启外科手术模式：修复数据库元数据，不触碰 Faiss 索引！")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 找出库里所有独特的文件路径
    cursor.execute("SELECT DISTINCT file_path FROM songs")
    rows = cursor.fetchall()

    print(f"🔍 找到 {len(rows)} 首独立歌曲，正在提取真实 MP3 标签...")

    success_count = 0
    for row in rows:
        file_path = row[0]
        try:
            # 使用 tinytag 读取真实的 MP3 内部元数据
            tag = TinyTag.get(file_path)

            # 如果读到了真实的 title 就用真实的，否则兜底用文件名
            real_title = tag.title if tag.title else os.path.basename(file_path)
            real_artist = tag.artist if tag.artist else "Unknown Artist"

            # 执行更新手术
            cursor.execute(
                "UPDATE songs SET title = ?, artist = ? WHERE file_path = ?",
                (real_title, real_artist, file_path)
            )
            success_count += 1

        except Exception as e:
            # 万一有坏文件，静默跳过
            pass

    conn.commit()
    conn.close()
    print(f"🎉 修复大功告成！成功找回 {success_count} 首歌的真实姓名！")
    print("👉 现在去刷新你的网页，看看《Food》是不是回来了！")


if __name__ == "__main__":
    fix_metadata()