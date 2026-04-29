import pytest
from fastapi.testclient import TestClient


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "index_loaded" in data
    assert "model_loaded" in data


def test_register_login_flow(client: TestClient):
    resp = client.post("/register", json={"username": "alice", "password": "pass1234"})
    assert resp.status_code == 200
    assert resp.json()["code"] == 200

    resp = client.post("/login", json={"username": "alice", "password": "pass1234"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["code"] == 200
    assert "access_token" in data["data"]
    assert "music_app_token" in resp.cookies

    resp = client.post("/login", json={"username": "alice", "password": "wrong"})
    assert resp.json()["code"] == 400


def test_register_duplicate(client: TestClient):
    client.post("/register", json={"username": "bob", "password": "pass1234"})
    resp = client.post("/register", json={"username": "bob", "password": "pass1234"})
    assert resp.json()["code"] == 400


def test_favorites_requires_auth(client: TestClient):
    resp = client.post("/favorite/toggle", json={
        "file_path": "/test.mp3", "title": "test", "artist": "unknown"
    })
    assert resp.status_code == 401


def test_history_requires_auth(client: TestClient):
    resp = client.get("/history")
    assert resp.status_code == 401


def test_rate_limit(client: TestClient):
    for _ in range(110):
        client.post("/login", json={"username": "nobody", "password": "x"})
    resp = client.post("/login", json={"username": "nobody", "password": "x"})
    assert resp.status_code == 429


def test_search_no_file(client: TestClient):
    resp = client.post("/search")
    assert resp.status_code == 422


def test_upload_rejected_bad_extension(client: TestClient):
    resp = client.post("/search", files={
        "file": ("test.txt", b"not audio", "text/plain")
    })
    assert resp.status_code == 400


def test_upload_rejected_too_large(client: TestClient):
    import io
    large = b"\x00" * (51 * 1024 * 1024)
    resp = client.post("/search", files={
        "file": ("large.wav", io.BytesIO(large), "audio/wav")
    })
    assert resp.status_code == 413


def test_upload_uses_server_generated_temp_path(client: TestClient, monkeypatch):
    import os
    import tempfile

    seen = {}

    def fake_search_music(temp_file_path, index, model):
        seen["path"] = temp_file_path
        assert os.path.exists(temp_file_path)
        return []

    import main
    monkeypatch.setattr(main, "search_music", fake_search_music)

    resp = client.post("/search", files={
        "file": ("../../owned.wav", b"fake wav data", "audio/wav")
    })

    assert resp.status_code == 200
    assert resp.json()["code"] == 200
    assert os.path.dirname(seen["path"]) == tempfile.gettempdir()
    assert os.path.basename(seen["path"]).startswith("music_upload_")
    assert not os.path.exists(seen["path"])
    assert not os.path.exists("owned.wav")
