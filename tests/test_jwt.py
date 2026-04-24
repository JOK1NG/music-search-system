import pytest
import jwt
from config import JWT_SECRET, JWT_ALGORITHM


def test_jwt_create_and_verify():
    from main import create_access_token, get_current_user
    token = create_access_token(user_id=1, username="testuser")
    assert token is not None
    decoded = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    assert decoded["sub"] == "1"
    assert decoded["username"] == "testuser"


def test_jwt_expired():
    import time
    from datetime import datetime, timedelta, timezone
    payload = {
        "sub": "1",
        "username": "test",
        "exp": datetime.now(timezone.utc) - timedelta(days=1),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    with pytest.raises(jwt.ExpiredSignatureError):
        jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
