import os
import sys
import tempfile
import pytest

os.environ["JWT_SECRET"] = "test-secret-key-32chars-minimum-xxx"
os.environ["INDEX_PATH"] = os.path.join(tempfile.gettempdir(), "test_music_hash.index")
os.environ["CORS_ORIGINS"] = "http://localhost:8000"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["RATE_LIMIT_AUTH"] = "10"
os.environ["RATE_LIMIT_GENERAL"] = "60"
os.environ["MAX_UPLOAD_SIZE"] = "52428800"


@pytest.fixture
def client():
    db_fd, db_path = tempfile.mkstemp(suffix=".db", prefix="test_music_hash_")
    os.close(db_fd)
    os.environ["DB_PATH"] = db_path

    from main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c

    try:
        os.unlink(db_path)
    except OSError:
        pass
