import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import shutil
import time
import uuid
import contextlib
from collections import defaultdict

import uvicorn
import sqlite3
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel
import bcrypt
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config import (
    DB_PATH, INDEX_PATH, MODEL_PATH,
    JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRE_DAYS,
    CORS_ORIGIN_LIST,
    RATE_LIMIT_SEARCH, RATE_LIMIT_AUTH, RATE_LIMIT_GENERAL,
    MAX_UPLOAD_SIZE, ALLOWED_AUDIO_EXTENSIONS,
    TOP_N_RESULTS,
)
from search import load_system, search_music

# --- 日志 ---
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

# --- JWT ---
def create_access_token(user_id: int, username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {"sub": str(user_id), "username": username, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(request: Request) -> dict:
    token = request.cookies.get("music_app_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
    if not token:
        raise HTTPException(status_code=401, detail="未登录，请先登录后再操作")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"user_id": int(payload["sub"]), "username": payload["username"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的身份令牌")


# --- 速率限制 (in-memory sliding window) ---
class RateLimiter:
    def __init__(self):
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._limits: dict[str, int] = {}
        self._cleanup_ts = time.time()

    def configure(self, path: str, rpm: int):
        self._limits[path] = rpm

    def is_allowed(self, path: str) -> bool:
        limit = self._limits.get(path, self._limits.get("general", 60))
        now = time.time()
        window = self._windows[path]
        window[:] = [t for t in window if now - t < 60]
        if len(window) >= limit:
            return False
        window.append(now)
        return True


rate_limiter = RateLimiter()
rate_limiter.configure("/search", RATE_LIMIT_SEARCH)
rate_limiter.configure("/login", RATE_LIMIT_AUTH)
rate_limiter.configure("/register", RATE_LIMIT_AUTH)
rate_limiter.configure("general", RATE_LIMIT_GENERAL)

# --- SQLite 连接管理器 ---
@contextlib.contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


# --- 系统状态 ---
system_state = {
    "index_loaded": False,
    "model_loaded": False,
    "index": None,
    "model": None,
}

# --- FastAPI ---
app = FastAPI(title="音乐音频相似度检索系统 API")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("./index.html")


@app.get("/favicon.svg", include_in_schema=False)
async def serve_favicon():
    return FileResponse("./favicon.svg")


app.mount("/audio", StaticFiles(directory="./data/raw"), name="audio")


def init_user_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                title TEXT,
                artist TEXT,
                UNIQUE(user_id, file_path),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                query_filename TEXT,
                top_title TEXT,
                top_artist TEXT,
                top_audio_url TEXT,
                top_file_path TEXT,
                searched_at TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_favorites_user_id ON favorites(user_id)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_history_user_id ON search_history(user_id)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_history_searched_at ON search_history(searched_at)
        ''')
        conn.commit()


init_user_db()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGIN_LIST if CORS_ORIGIN_LIST else ["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 速率限制中间件 ---
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if not rate_limiter.is_allowed(path):
        logger.warning(f"Rate limit exceeded for {request.client.host} on {path}")
        return JSONResponse(
            status_code=429,
            content={"code": 429, "message": "请求过于频繁，请稍后再试"}
        )
    return await call_next(request)


# --- CSP 安全头中间件 ---
CSP_POLICY = os.environ.get("CSP_POLICY", "; ".join([
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' https://unpkg.com",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    "connect-src 'self'",
    "media-src 'self'",
    "img-src 'self' data: blob:",
    "worker-src blob:",
]))


@app.middleware("http")
async def csp_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = CSP_POLICY
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


# --- 健康检查 ---
@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "ok",
        "index_loaded": system_state["index_loaded"],
        "model_loaded": system_state["model_loaded"],
        "index_size": system_state["index"].ntotal if system_state["index"] else 0,
    }


# --- 鉴权端点 ---
class UserReq(BaseModel):
    username: str
    password: str


@app.post("/logout")
async def logout(request: Request, response: Response):
    response.delete_cookie(key="music_app_token")
    return {"code": 200, "message": "已登出"}


@app.post("/register")
async def register(user: UserReq, response: Response):
    with get_db() as conn:
        try:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(user.password.encode('utf-8'), salt).decode('utf-8')
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (user.username, hashed)
            )
            conn.commit()
            return {"code": 200, "message": "注册成功！欢迎加入！"}
        except sqlite3.IntegrityError:
            return {"code": 400, "message": "用户名已经被抢注啦，换一个吧！"}


@app.post("/login")
async def login(user: UserReq, response: Response):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE username=?", (user.username,)
        ).fetchone()

    if row is None:
        return {"code": 400, "message": "用户名不存在哦！"}

    user_id, hashed_password = row["id"], row["password_hash"]

    if not bcrypt.checkpw(user.password.encode('utf-8'), hashed_password.encode('utf-8')):
        return {"code": 400, "message": "密码错误，再想想？"}

    access_token = create_access_token(user_id=user_id, username=user.username)

    response.set_cookie(
        key="music_app_token",
        value=access_token,
        httponly=True,
        secure=False,  # 生产改为 True
        samesite="strict",
        max_age=JWT_EXPIRE_DAYS * 86400,
    )

    return {
        "code": 200,
        "message": "登录成功！",
        "data": {"user_id": user_id, "username": user.username, "access_token": access_token}
    }


# --- 收藏 ---
class FavoriteReq(BaseModel):
    file_path: str
    title: str
    artist: str


@app.post("/favorite/toggle")
async def toggle_favorite(req: FavoriteReq, request: Request):
    current_user = get_current_user(request)
    user_id = current_user["user_id"]

    with get_db() as conn:
        row = conn.execute(
            "SELECT id FROM favorites WHERE user_id=? AND file_path=?",
            (user_id, req.file_path)
        ).fetchone()

        if row:
            conn.execute("DELETE FROM favorites WHERE id=?", (row["id"],))
            msg, status = "已取消收藏", "removed"
        else:
            conn.execute(
                "INSERT INTO favorites (user_id, file_path, title, artist) VALUES (?, ?, ?, ?)",
                (user_id, req.file_path, req.title, req.artist)
            )
            msg, status = "收藏成功！", "added"

        conn.commit()
        return {"code": 200, "message": msg, "data": {"status": status}}


@app.get("/favorites/{user_id}")
async def get_favorites(user_id: int, request: Request):
    current_user = get_current_user(request)
    if current_user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="无权访问他人的收藏夹")

    with get_db() as conn:
        rows = conn.execute(
            "SELECT title, artist, file_path FROM favorites WHERE user_id=?", (user_id,)
        ).fetchall()

    results = []
    for row in rows:
        normalized_path = row["file_path"].replace("\\", "/")
        audio_url = "/audio/" + normalized_path.split("raw/")[-1] if "raw/" in normalized_path else ""
        results.append({
            "title": row["title"],
            "artist": row["artist"],
            "file_path": row["file_path"],
            "audio_url": audio_url
        })
    return {"code": 200, "data": results}


# --- 上传校验 ---
ALLOWED_MIME_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/flac", "audio/ogg", "audio/mp4", "audio/aac",
    "audio/x-m4a", "application/octet-stream",
}


def validate_upload(file: UploadFile):
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    allowed_exts = [e.strip().lower() for e in ALLOWED_AUDIO_EXTENSIONS]
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，支持: {', '.join(allowed_exts)}"
        )
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"可疑 MIME 类型: {file.content_type}，仍允许继续处理")


@app.post("/search")
async def api_search_music(file: UploadFile = File(...), request: Request = None):
    validate_upload(file)

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        logger.warning(f"上传文件过大: {len(content)} bytes from {request.client.host if request else '?'}")
        raise HTTPException(status_code=413, detail="文件大小超出限制")

    safe_suffix = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
    temp_file_path = f"{uuid.uuid4().hex}{safe_suffix}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(content)

    try:
        results = search_music(temp_file_path, system_state["index"], system_state["model"])

        user = None
        try:
            user = get_current_user(request)
        except HTTPException:
            pass

        if user:
            try:
                top = results[0] if results else None
                with get_db() as conn:
                    conn.execute(
                        "INSERT INTO search_history (user_id, query_filename, top_title, top_artist, top_audio_url, top_file_path) VALUES (?,?,?,?,?,?)",
                        (
                            user["user_id"],
                            file.filename,
                            top["title"] if top else None,
                            top["artist"] if top else None,
                            top["audio_url"] if top else None,
                            top["file_path"] if top else None,
                        )
                    )
                    conn.commit()
            except Exception:
                logger.exception("搜索历史记录写入失败")

        return {"code": 200, "message": "检索成功", "data": results}

    except Exception as e:
        logger.exception("音频检索失败")
        return {"code": 500, "message": f"处理音频时发生错误: {str(e)}", "data": []}
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                logger.warning(f"临时文件删除失败: {temp_file_path}")


# --- 搜索历史 ---
@app.get("/history")
async def get_history(request: Request):
    user = get_current_user(request)
    with get_db() as conn:
        rows = conn.execute(
            """SELECT query_filename, top_title, top_artist, top_audio_url, top_file_path, searched_at
               FROM search_history WHERE user_id=?
               ORDER BY id DESC LIMIT 50""",
            (user["user_id"],)
        ).fetchall()

    return {
        "code": 200,
        "data": [
            {
                "query_filename": r["query_filename"],
                "top_title": r["top_title"],
                "top_artist": r["top_artist"],
                "top_audio_url": r["top_audio_url"],
                "top_file_path": r["top_file_path"],
                "searched_at": r["searched_at"],
            } for r in rows
        ]
    }


@app.delete("/history")
async def clear_history(request: Request):
    user = get_current_user(request)
    with get_db() as conn:
        conn.execute("DELETE FROM search_history WHERE user_id=?", (user["user_id"],))
        conn.commit()
    return {"code": 200, "message": "搜索历史已清空"}


# --- 优雅关闭 ---
import atexit

def cleanup():
    logger.info("正在执行清理操作...")
    if system_state["index"] is not None:
        del system_state["index"]
        system_state["index"] = None
    if system_state["model"] is not None:
        del system_state["model"]
        system_state["model"] = None
    # 清理残留临时文件
    import glob as _glob
    for f in _glob.glob("[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]" + "*"):
        try:
            os.remove(f)
        except OSError:
            pass
    logger.info("清理完成")

atexit.register(cleanup)

if __name__ == "__main__":
    logger.info("正在初始化全局检索系统...")
    system_state["index"], system_state["model"] = load_system()
    system_state["index_loaded"] = system_state["index"] is not None
    system_state["model_loaded"] = system_state["model"] is not None
    logger.info("Web 服务器后台准备就绪！")
    uvicorn.run(app, host="127.0.0.1", port=8000)
