"""
重建 music_hash.db 的 songs 表，使其与现有的 music_hash.index 完美对齐。

- 不跑 CNN 推理，只复刻 batch_extract.py 的「文件遍历 + 时长切片」逻辑
- 只写 SQLite，不写 Faiss，零风险复用当前索引
- 全程在临时表 songs_tmp 上累积，校验 COUNT(songs_tmp) == faiss.ntotal 通过后
  才用单事务把旧 songs 替换成新表；任何异常或数量不一致都直接 DROP 临时表，
  原始 songs 表保持原样，绝不残留半成品。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import _utf8_console  # noqa: F401  强制 stdout/stderr 走 UTF-8，规避 Windows cp936 重定向丢字符（详见模块内注释）

import sqlite3
import time

import faiss
import librosa
from tinytag import TinyTag

RAW_PATH = "./data/raw"
DB_PATH = "music_hash.db"
INDEX_PATH = "music_hash.index"


def _create_tmp_table(cur):
    cur.execute("DROP TABLE IF EXISTS songs_tmp")
    cur.execute(
        """CREATE TABLE songs_tmp (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            faiss_id INTEGER,
            title TEXT,
            artist TEXT,
            file_path TEXT,
            offset REAL
        )"""
    )


def _swap_in_tmp_table(conn, cur):
    """把 songs_tmp 原子替换 songs：要么整体成功，要么旧表完整保留。"""
    cur.execute("BEGIN IMMEDIATE")
    try:
        cur.execute("DROP TABLE IF EXISTS songs")
        cur.execute("ALTER TABLE songs_tmp RENAME TO songs")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_songs_faiss_id ON songs(faiss_id)")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def main():
    index = faiss.read_index_binary(INDEX_PATH)
    expected = index.ntotal
    print(f"🎯 目标子指纹总数 (faiss ntotal) = {expected}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    _create_tmp_table(cur)
    conn.commit()

    music_files = []
    for root, _dirs, files in os.walk(RAW_PATH):
        for f in files:
            if f.lower().endswith((".mp3", ".wav")):
                music_files.append(os.path.join(root, f))

    print(f"🎵 共扫描到 {len(music_files)} 首音频。开始重建临时 songs_tmp 表…")

    faiss_id = 0
    t0 = time.time()
    failures = 0
    batch = []

    try:
        for i, file_path in enumerate(music_files):
            try:
                try:
                    tag = TinyTag.get(file_path)
                    title = tag.title if tag.title else os.path.basename(file_path)
                    artist = tag.artist if tag.artist else "Unknown Artist"
                except Exception:
                    title = os.path.basename(file_path)
                    artist = "Unknown"

                # 与 batch_extract.py 完全一致的时长推断
                y, sr = librosa.load(file_path, sr=22050, duration=30.0)
                total_duration = len(y) / sr
                if total_duration < 5.0:
                    total_duration = 5.0

                for start_sec in range(0, int(total_duration - 4)):
                    batch.append((faiss_id, title, artist, file_path, float(start_sec)))
                    faiss_id += 1

                if (i + 1) % 200 == 0:
                    cur.executemany(
                        "INSERT INTO songs_tmp (faiss_id, title, artist, file_path, offset) VALUES (?,?,?,?,?)",
                        batch,
                    )
                    batch.clear()
                    conn.commit()
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(music_files) - (i + 1)) / rate if rate > 0 else 0
                    print(
                        f"⏳ [{i+1}/{len(music_files)}] 已生成 {faiss_id} 条 | "
                        f"{rate:.1f} 首/秒 | ETA {eta/60:.1f} 分钟"
                    )
            except Exception:
                failures += 1

        if batch:
            cur.executemany(
                "INSERT INTO songs_tmp (faiss_id, title, artist, file_path, offset) VALUES (?,?,?,?,?)",
                batch,
            )
            batch.clear()
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM songs_tmp")
        actual = cur.fetchone()[0]
        print(
            f"📊 songs_tmp 行数 = {actual}，faiss ntotal = {expected}，"
            f"解码失败 {failures} 首"
        )

        if actual != expected:
            print(
                "❌ 数量不一致！临时表保留以供调试，旧 songs 表未受影响。\n"
                "   请改跑 batch_extract.py 完整重建模型 + 索引 + 元数据。"
            )
            return 1

        _swap_in_tmp_table(conn, cur)
        print("✅ songs 表重建成功，与 faiss 索引 1:1 对齐。")
        return 0

    except Exception as e:
        # 任何异常都丢弃临时表，不污染旧 songs
        try:
            cur.execute("DROP TABLE IF EXISTS songs_tmp")
            conn.commit()
        except Exception:
            pass
        print(f"❌ 重建过程中异常退出，已回滚临时表，旧 songs 保持原样：{e}")
        return 2

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
