import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 修复 macOS ARM64 OpenMP 冲突崩溃
os.environ["OMP_NUM_THREADS"] = "1"            # 防止 PyTorch 在 uvicorn 线程中引发 SIGSEGV
os.environ["MKL_NUM_THREADS"] = "1"            # 同上，限制 MKL 线程

import tempfile
import time
import uvicorn
import sqlite3
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional
from collections import defaultdict
from pydantic import BaseModel
import bcrypt
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request, Response, Cookie
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 导入你刚刚写好的终极版搜索器
from search import load_system, search_music

# ==========================================
# 🔐 JWT 配置区
# ==========================================
# 生产环境部署时，请通过环境变量 JWT_SECRET 设置一个强随机密钥：
#   export JWT_SECRET="your-super-secret-key-here"
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET or JWT_SECRET == "vectortune-dev-secret-change-in-production":
    raise RuntimeError(
        "JWT_SECRET 环境变量未设置或仍为旧默认值。"
        "请配置强随机密钥，例如: export JWT_SECRET=$(openssl rand -hex 32)"
    )
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 7  # 令牌有效期：7 天
DB_PATH = os.environ.get("DB_PATH", "music_hash.db")
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 50 * 1024 * 1024))
ALLOWED_AUDIO_EXTENSIONS = {
    ext.strip().lower()
    for ext in os.environ.get("ALLOWED_AUDIO_EXTENSIONS", ".mp3,.wav,.flac,.ogg,.m4a,.aac").split(",")
    if ext.strip()
}
CORS_ORIGIN_LIST = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    if origin.strip()
]
RATE_LIMIT_SEARCH = int(os.environ.get("RATE_LIMIT_SEARCH", 30))
RATE_LIMIT_AUTH = int(os.environ.get("RATE_LIMIT_AUTH", 100))
RATE_LIMIT_GENERAL = int(os.environ.get("RATE_LIMIT_GENERAL", 300))


class RateLimiter:
    def __init__(self):
        self._windows = defaultdict(list)

    def is_allowed(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        now = time.time()
        window = self._windows[key]
        window[:] = [t for t in window if now - t < window_seconds]
        if len(window) >= limit:
            return False
        window.append(now)
        return True


rate_limiter = RateLimiter()


def get_client_id(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


def create_access_token(user_id: int, username: str) -> str:
    """生成 JWT 访问令牌"""
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": expire
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(authorization: Optional[str] = None, music_app_token: Optional[str] = None) -> dict:
    """从请求头提取并验证 JWT，返回用户信息字典"""
    token = music_app_token
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
    if not token:
        raise HTTPException(status_code=401, detail="未登录，请先登录后再操作")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"user_id": int(payload["sub"]), "username": payload["username"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的身份令牌")

app = FastAPI(title="音乐音频相似度检索系统 API")


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if path == "/search":
        limit = RATE_LIMIT_SEARCH
    elif path in {"/login", "/register"}:
        limit = RATE_LIMIT_AUTH
    else:
        limit = RATE_LIMIT_GENERAL

    key = f"{get_client_id(request)}:{path}"
    if not rate_limiter.is_allowed(key, limit):
        return JSONResponse(
            status_code=429,
            content={"code": 429, "message": "请求过于频繁，请稍后再试"}
        )
    return await call_next(request)

# 🌟 前端入口：GET / 直接返回 index.html（让 window.location.origin 能正确推断 API 地址）
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("./index.html")

@app.get("/favicon.svg", include_in_schema=False)
async def serve_favicon():
    return FileResponse("./favicon.svg")

# 🌟 把本地的 ./data/raw 文件夹，挂载到网址的 /audio 路径下
app.mount("/audio", StaticFiles(directory="./data/raw"), name="audio")


@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "ok",
        "index_loaded": INDEX is not None,
        "model_loaded": MODEL is not None,
        "index_size": INDEX.ntotal if INDEX is not None else 0,
    }

# ==========================================
# 🌟 用户系统配置区
# ==========================================

# 接收前端传来的账号密码的格式模型
class UserReq(BaseModel):
    username: str
    password: str
# 数据库扩容：创建用户表和收藏表
def init_user_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 1. 创建用户表 (username 必须唯一)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    # 2. 创建收藏表 (记录哪个用户收藏了哪首歌)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            file_path TEXT NOT NULL,
            title TEXT,
            artist TEXT,
            UNIQUE(user_id, file_path) 
        )
    ''')
    # 3. 创建搜索历史表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            query_filename TEXT,
            top_title TEXT,
            top_artist TEXT,
            top_audio_url TEXT,
            top_file_path TEXT,
            searched_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    ''')
    conn.commit()
    conn.close()
# 每次启动服务器时自动建表
init_user_db()

# 配置 CORS 跨域（非常关键！否则后续 Vue 前端会被浏览器拦截报错）
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGIN_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⏳ 正在初始化全局检索系统...")
# 在服务器启动时，只加载一次模型和 Faiss 索引到内存中
# 这样用户搜索时就不会卡顿了
INDEX, MODEL = load_system()
print("✅ Web 服务器后台准备就绪！")


# ==========================================
# 🌟 注册接口 (纯净 bcrypt 版)
# ==========================================
@app.post("/register")
async def register(user: UserReq):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # 🌟 修改点：使用原生 bcrypt 把密码碾碎加盐，并转成字符串存入数据库
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), salt).decode('utf-8')

        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (user.username, hashed_password)
        )
        conn.commit()
        return {"code": 200, "message": "注册成功！欢迎加入！"}
    except sqlite3.IntegrityError:
        return {"code": 400, "message": "哎呀，用户名已经被抢注啦，换一个吧！"}
    finally:
        conn.close()


# ==========================================
# 🌟 登录接口 (纯净 bcrypt 版)
# ==========================================
@app.post("/login")
async def login(user: UserReq, response: Response):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id, password_hash FROM users WHERE username=?", (user.username,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"code": 400, "message": "用户名不存在哦！"}

    user_id, hashed_password = row

    # 🌟 修改点：使用原生 bcrypt 核对用户输入的密码和数据库里的乱码是否匹配
    if not bcrypt.checkpw(user.password.encode('utf-8'), hashed_password.encode('utf-8')):
        return {"code": 400, "message": "密码错误，再想想？"}

    # 🌟 生成 JWT 令牌，有效期 7 天
    access_token = create_access_token(user_id=user_id, username=user.username)
    response.set_cookie(
        key="music_app_token",
        value=access_token,
        httponly=True,
        secure=os.environ.get("COOKIE_SECURE", "0") == "1",
        samesite="strict",
        max_age=JWT_EXPIRE_DAYS * 86400,
    )

    return {
        "code": 200,
        "message": "登录成功！",
        "data": {
            "user_id": user_id,
            "username": user.username,
            "access_token": access_token
        }
    }


# ==========================================
# 🌟 收藏系统专用数据模型 (user_id 从 JWT 提取，不再信任前端传值)
# ==========================================
class FavoriteReq(BaseModel):
    file_path: str
    title: str
    artist: str


# ==========================================
# 🌟 收藏 / 取消收藏 (JWT 鉴权版)
# ==========================================
@app.post("/favorite/toggle")
async def toggle_favorite(
    req: FavoriteReq,
    authorization: Optional[str] = Header(None),
    music_app_token: Optional[str] = Cookie(None),
):
    # 从 JWT 中安全提取 user_id，不信任前端传来的数值
    current_user = get_current_user(authorization, music_app_token)
    user_id = current_user["user_id"]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 先查一下这首歌这个用户是不是已经收藏过了
    cursor.execute("SELECT id FROM favorites WHERE user_id=? AND file_path=?", (user_id, req.file_path))
    row = cursor.fetchone()

    if row:
        # 已经收藏了？那就取消收藏（删除记录）
        cursor.execute("DELETE FROM favorites WHERE id=?", (row[0],))
        msg = "已取消收藏 💔"
        status = "removed"
    else:
        # 没收藏？那就加进收藏夹
        cursor.execute("INSERT INTO favorites (user_id, file_path, title, artist) VALUES (?, ?, ?, ?)",
                       (user_id, req.file_path, req.title, req.artist))
        msg = "收藏成功！❤️"
        status = "added"

    conn.commit()
    conn.close()
    return {"code": 200, "message": msg, "data": {"status": status}}


# ==========================================
# 🌟 获取用户的专属收藏夹列表 (JWT 鉴权版)
# ==========================================
@app.get("/favorites/{user_id}")
async def get_favorites(
    user_id: int,
    authorization: Optional[str] = Header(None),
    music_app_token: Optional[str] = Cookie(None),
):
    # 验证 JWT，并确保只能查自己的收藏夹
    current_user = get_current_user(authorization, music_app_token)
    if current_user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="无权访问他人的收藏夹")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 只查属于这个用户的歌
    cursor.execute("SELECT title, artist, file_path FROM favorites WHERE user_id=?", (user_id,))
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        title, artist, file_path = row
        # 这里同样需要把本地路径翻译成前端能播放的 URL
        normalized_path = file_path.replace("\\", "/")
        audio_url = "/audio/" + normalized_path.split("raw/")[-1] if "raw/" in normalized_path else ""

        results.append({
            "title": title,
            "artist": artist,
            "file_path": file_path,
            "audio_url": audio_url
        })

    return {"code": 200, "data": results}


@app.post("/logout")
async def logout(response: Response):
    response.delete_cookie(key="music_app_token")
    return {"code": 200, "message": "已登出"}


def validate_upload(file: UploadFile):
    original_name = os.path.basename(file.filename or "")
    suffix = os.path.splitext(original_name)[1].lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {suffix or '无后缀'}"
        )
    return original_name, suffix

@app.post("/search")
async def api_search_music(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),   # 可选，不强制要求登录
    music_app_token: Optional[str] = Cookie(None),
):
    """
    接收前端上传的音频文件，进行听歌识曲。
    已登录用户的搜索记录会自动写入历史表。
    """
    # 1. 将用户上传的音频保存为服务端生成的临时文件名，避免路径穿越与覆盖本地文件
    original_name, safe_suffix = validate_upload(file)
    bytes_written = 0

    with tempfile.NamedTemporaryFile(delete=False, prefix="music_upload_", suffix=safe_suffix) as buffer:
        temp_file_path = buffer.name
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_SIZE:
                buffer.close()
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                raise HTTPException(status_code=413, detail="文件大小超出限制")
            buffer.write(chunk)

    try:
        # 2. 调用核心搜索逻辑，默认返回最相似的 5 首歌
        results = search_music(temp_file_path, INDEX, MODEL)

        # 🌟 已登录用户：静默记录搜索历史（失败不影响检索结果返回）
        if (authorization and authorization.startswith("Bearer ")) or music_app_token:
            try:
                user = get_current_user(authorization, music_app_token)
                top = results[0] if results else None
                conn = sqlite3.connect(DB_PATH)
                conn.execute(
                    "INSERT INTO search_history (user_id, query_filename, top_title, top_artist, top_audio_url, top_file_path) VALUES (?,?,?,?,?,?)",
                    (
                        user["user_id"],
                        original_name,
                        top["title"]     if top else None,
                        top["artist"]    if top else None,
                        top["audio_url"] if top else None,
                        top["file_path"] if top else None,
                    )
                )
                conn.commit()
                conn.close()
            except Exception:
                pass  # 静默失败，不影响主流程

        return {"code": 200, "message": "检索成功", "data": results}

    except Exception as e:
        return {"code": 500, "message": f"处理音频时发生错误: {str(e)}", "data": []}
    finally:
        # 3. 无论成功与否，都要把临时文件删掉，防止服务器硬盘被撑爆
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# ==========================================
# 🌟 搜索历史接口（JWT 鉴权）
# ==========================================
@app.get("/history")
async def get_history(
    authorization: Optional[str] = Header(None),
    music_app_token: Optional[str] = Cookie(None),
):
    """获取当前用户最近 50 条搜索历史，按时间倒序"""
    user = get_current_user(authorization, music_app_token)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT query_filename, top_title, top_artist, top_audio_url, top_file_path, searched_at
           FROM search_history WHERE user_id=?
           ORDER BY id DESC LIMIT 50""",
        (user["user_id"],)
    )
    rows = cursor.fetchall()
    conn.close()
    return {
        "code": 200,
        "data": [
            {
                "query_filename": r[0],
                "top_title":      r[1],
                "top_artist":     r[2],
                "top_audio_url":  r[3],
                "top_file_path":  r[4],
                "searched_at":    r[5],
            } for r in rows
        ]
    }


@app.delete("/history")
async def clear_history(
    authorization: Optional[str] = Header(None),
    music_app_token: Optional[str] = Cookie(None),
):
    """清空当前用户的全部搜索历史"""
    user = get_current_user(authorization, music_app_token)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM search_history WHERE user_id=?", (user["user_id"],))
    conn.commit()
    conn.close()
    return {"code": 200, "message": "搜索历史已清空"}


if __name__ == "__main__":
    # 启动服务器，运行在本地 8000 端口
    uvicorn.run(app, host="127.0.0.1", port=8000)
