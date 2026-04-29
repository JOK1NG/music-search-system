import jwt
import os
import pytest
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRE_DAYS


def _create_token(user_id, username):
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {"sub": str(user_id), "username": username, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def test_jwt_create_and_verify():
    token = _create_token(user_id=1, username="testuser")
    assert token is not None
    decoded = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    assert decoded["sub"] == "1"
    assert decoded["username"] == "testuser"


def test_jwt_expired():
    payload = {
        "sub": "1",
        "username": "test",
        "exp": datetime.now(timezone.utc) - timedelta(days=1),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    with pytest.raises(jwt.ExpiredSignatureError):
        jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def test_jwt_secret_is_required_for_import():
    env = os.environ.copy()
    env.pop("JWT_SECRET", None)

    result = subprocess.run(
        [sys.executable, "-c", "import main"],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "JWT_SECRET" in result.stderr
