import sqlite3

import numpy as np
import torch

import search
from config import SAMPLE_RATE, WINDOW_DURATION


class DummyModel:
    def parameters(self):
        return iter([torch.zeros(1)])


class DummyIndex:
    def search(self, query_packed, k):
        assert k == 4
        distances = np.array([[7, 3, 5, 0]], dtype=np.int64)
        faiss_ids = np.array([[10, 11, 12, -1]], dtype=np.int64)
        return distances, faiss_ids


def test_search_music_batches_metadata_lookup_and_preserves_result_shape(tmp_path, monkeypatch):
    db_path = tmp_path / "music_hash.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE songs (
            faiss_id INTEGER,
            title TEXT,
            artist TEXT,
            file_path TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO songs (faiss_id, title, artist, file_path) VALUES (?, ?, ?, ?)",
        [
            (10, "Song A old", "Artist A", "data/raw/song-a.mp3"),
            (11, "Song A best", "Artist A", "data/raw/song-a.mp3"),
            (12, "Song B", "Artist B", r"data\raw\nested\song-b.wav"),
        ],
    )
    conn.commit()
    conn.close()

    select_statements = []
    real_connect = sqlite3.connect

    class CountingCursor:
        def __init__(self, cursor):
            self._cursor = cursor

        def execute(self, sql, params=()):
            if sql.lstrip().upper().startswith("SELECT"):
                select_statements.append(sql)
            return self._cursor.execute(sql, params)

        def fetchall(self):
            return self._cursor.fetchall()

    class CountingConnection:
        def __init__(self, path):
            self._conn = real_connect(path)

        def cursor(self):
            return CountingCursor(self._conn.cursor())

        def close(self):
            self._conn.close()

    monkeypatch.setattr(search, "DB_PATH", str(db_path))
    monkeypatch.setattr(search.sqlite3, "connect", CountingConnection)
    monkeypatch.setattr(
        search,
        "load_audio",
        lambda _path, duration=None: (np.ones(int(SAMPLE_RATE * WINDOW_DURATION), dtype=np.float32), SAMPLE_RATE),
    )
    monkeypatch.setattr(
        search,
        "slice_best_windows",
        lambda y, window_duration, n_windows, sr: [(1.0, y, 0)],
    )
    monkeypatch.setattr(
        search,
        "extract_hash_from_window",
        lambda y_window, model, device, sr: np.zeros((1, 16), dtype=np.uint8),
    )

    results = search.search_music("query.wav", DummyIndex(), DummyModel(), top_k=4, top_n_results=2)

    assert len(select_statements) == 1
    assert " IN " in select_statements[0]
    assert results == [
        {
            "rank": 1,
            "title": "Song A best",
            "artist": "Artist A",
            "distance": 3,
            "file_path": "data/raw/song-a.mp3",
            "audio_url": "/audio/song-a.mp3",
        },
        {
            "rank": 2,
            "title": "Song B",
            "artist": "Artist B",
            "distance": 5,
            "file_path": r"data\raw\nested\song-b.wav",
            "audio_url": "/audio/nested/song-b.wav",
        },
    ]
