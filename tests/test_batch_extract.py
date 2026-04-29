import sqlite3
from types import SimpleNamespace

import numpy as np


SONGS_SCHEMA = """
    CREATE TABLE songs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        faiss_id INTEGER,
        title TEXT,
        artist TEXT,
        file_path TEXT,
        offset REAL
    )
"""


class FakeModel:
    def to(self, device):
        return self

    def load_state_dict(self, state):
        self.state = state

    def eval(self):
        self.evaluated = True


class FakeIndex:
    def __init__(self, bits):
        self.bits = bits
        self.items = []

    def add(self, packed):
        self.items.append(packed)


def create_old_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute(SONGS_SCHEMA)
    conn.execute(
        "INSERT INTO songs (faiss_id, title, artist, file_path, offset) VALUES (?, ?, ?, ?, ?)",
        (99, "old-title", "old-artist", "old.mp3", 12.0),
    )
    conn.commit()
    conn.close()


def read_songs(db_path):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT faiss_id, title, artist, file_path, offset FROM songs ORDER BY faiss_id"
    ).fetchall()
    conn.close()
    return rows


def table_exists(db_path, table_name):
    conn = sqlite3.connect(db_path)
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    conn.close()
    return exists is not None


def patch_batch_extract(monkeypatch, tmp_path, load_audio):
    import batch_extract

    raw_path = tmp_path / "raw"
    db_path = tmp_path / "music_hash.db"
    index_path = tmp_path / "music_hash.index"
    raw_path.mkdir()

    monkeypatch.setattr(batch_extract, "RAW_PATH", str(raw_path))
    monkeypatch.setattr(batch_extract, "DB_PATH", str(db_path))
    monkeypatch.setattr(batch_extract, "INDEX_PATH", str(index_path))
    monkeypatch.setattr(batch_extract, "MODEL_PATH", str(tmp_path / "unused_model.pth"))
    monkeypatch.setattr(batch_extract, "AudioHashNet", FakeModel)
    monkeypatch.setattr(batch_extract.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(batch_extract.torch, "load", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(batch_extract.faiss, "IndexBinaryFlat", FakeIndex)

    def write_index_binary(index, path):
        with open(path, "w") as index_file:
            index_file.write(f"fingerprints={len(index.items)}")

    monkeypatch.setattr(batch_extract.faiss, "write_index_binary", write_index_binary)
    monkeypatch.setattr(
        batch_extract.TinyTag,
        "get",
        lambda path: SimpleNamespace(title=None, artist=None),
    )
    monkeypatch.setattr(batch_extract.preprocessing, "load_audio", load_audio)
    monkeypatch.setattr(
        batch_extract.preprocessing,
        "extract_hash_from_window",
        lambda y_window, model, device, sr: np.ones((1, 16), dtype=np.uint8),
    )

    return batch_extract, raw_path, db_path, index_path


def test_batch_rebuild_swaps_table_and_index_only_after_success(monkeypatch, tmp_path):
    def load_audio(file_path, duration=None):
        return np.ones(60, dtype=np.float32), 10

    batch_extract, raw_path, db_path, index_path = patch_batch_extract(
        monkeypatch, tmp_path, load_audio
    )
    create_old_db(db_path)
    index_path.write_text("old-index")
    (raw_path / "song.wav").write_bytes(b"fake")

    assert batch_extract.extract_features() is True

    rows = read_songs(db_path)
    assert len(rows) == 2
    assert rows[0][0] == 0
    assert rows[0][1] == "song.wav"
    assert rows[0][2] == "Unknown Artist"
    assert index_path.read_text() == "fingerprints=2"
    assert not table_exists(db_path, batch_extract.BUILD_TABLE)
    assert not index_path.with_name(index_path.name + ".tmp").exists()


def test_batch_rebuild_keeps_existing_outputs_when_any_file_fails(monkeypatch, tmp_path):
    def load_audio(file_path, duration=None):
        if file_path.endswith("bad.wav"):
            raise ValueError("decode failed")
        return np.ones(60, dtype=np.float32), 10

    batch_extract, raw_path, db_path, index_path = patch_batch_extract(
        monkeypatch, tmp_path, load_audio
    )
    create_old_db(db_path)
    index_path.write_text("old-index")
    (raw_path / "good.wav").write_bytes(b"fake")
    (raw_path / "bad.wav").write_bytes(b"fake")

    assert batch_extract.extract_features() is False

    assert read_songs(db_path) == [(99, "old-title", "old-artist", "old.mp3", 12.0)]
    assert index_path.read_text() == "old-index"
    assert not table_exists(db_path, batch_extract.BUILD_TABLE)
    assert not index_path.with_name(index_path.name + ".tmp").exists()


def test_batch_rebuild_keeps_existing_outputs_when_no_fingerprints(monkeypatch, tmp_path):
    def load_audio(file_path, duration=None):
        return np.ones(60, dtype=np.float32), 10

    batch_extract, raw_path, db_path, index_path = patch_batch_extract(
        monkeypatch, tmp_path, load_audio
    )
    create_old_db(db_path)
    index_path.write_text("old-index")
    (raw_path / "notes.txt").write_text("not music")

    assert batch_extract.extract_features() is False

    assert read_songs(db_path) == [(99, "old-title", "old-artist", "old.mp3", 12.0)]
    assert index_path.read_text() == "old-index"
    assert not table_exists(db_path, batch_extract.BUILD_TABLE)
    assert not index_path.with_name(index_path.name + ".tmp").exists()
