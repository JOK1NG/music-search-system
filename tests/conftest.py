import os
import tempfile
import pytest

os.environ["JWT_SECRET"] = "test-secret-key-32chars-minimum-xxx"
os.environ["CORS_ORIGINS"] = "http://localhost:8000"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["RATE_LIMIT_AUTH"] = "100"
os.environ["RATE_LIMIT_GENERAL"] = "100"
os.environ["MAX_UPLOAD_SIZE"] = "52428800"

_TEMP_DIR = tempfile.mkdtemp(prefix="test_music_")
_TEST_DB_PATH = os.path.join(_TEMP_DIR, "test.db")
os.environ["DB_PATH"] = _TEST_DB_PATH
os.environ["INDEX_PATH"] = os.path.join(_TEMP_DIR, "test.index")


@pytest.fixture(scope="module")
def client():
    from main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clear_cookies(client):
    client.cookies.clear()
