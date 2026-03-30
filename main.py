import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 修复 macOS ARM64 OpenMP 冲突崩溃
os.environ["OMP_NUM_THREADS"] = "1"            # 防止 PyTorch 在 uvicorn 线程中引发 SIGSEGV
os.environ["MKL_NUM_THREADS"] = "1"            # 同上，限制 MKL 线程

import shutil
import uvicorn
import sqlite3
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional
from pydantic import BaseModel
import bcrypt
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 导入你刚刚写好的终极版搜索器
from search import load_system, search_music

# ==========================================
# 🔐 JWT 配置区
# ==========================================
# 生产环境部署时，请通过环境变量 JWT_SECRET 设置一个强随机密钥：
#   export JWT_SECRET="your-super-secret-key-here"
JWT_SECRET = os.environ.get("JWT_SECRET", "vectortune-dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 7  # 令牌有效期：7 天


def create_access_token(user_id: int, username: str) -> str:
    """生成 JWT 访问令牌"""
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": expire
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """从请求头提取并验证 JWT，返回用户信息字典"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录，请先登录后再操作")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"user_id": int(payload["sub"]), "username": payload["username"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的身份令牌")

app = FastAPI(title="音乐音频相似度检索系统 API")

# 🌟 前端入口：GET / 直接返回 index.html（让 window.location.origin 能正确推断 API 地址）
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("./index.html")

@app.get("/favicon.svg", include_in_schema=False)
async def serve_favicon():
    return FileResponse("./favicon.svg")

# 🌟 把本地的 ./data/raw 文件夹，挂载到网址的 /audio 路径下
app.mount("/audio", StaticFiles(directory="./data/raw"), name="audio")

# ==========================================
# 🌟 用户系统配置区
# ==========================================

# 接收前端传来的账号密码的格式模型
class UserReq(BaseModel):
    username: str
    password: str
# 数据库扩容：创建用户表和收藏表
def init_user_db():
    conn = sqlite3.connect("music_hash.db")
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
    allow_origins=["*"],  # 允许所有前端来源访问
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
    conn = sqlite3.connect("music_hash.db")
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
async def login(user: UserReq):
    conn = sqlite3.connect("music_hash.db")
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
async def toggle_favorite(req: FavoriteReq, authorization: Optional[str] = Header(None)):
    # 从 JWT 中安全提取 user_id，不信任前端传来的数值
    current_user = get_current_user(authorization)
    user_id = current_user["user_id"]

    conn = sqlite3.connect("music_hash.db")
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
async def get_favorites(user_id: int, authorization: Optional[str] = Header(None)):
    # 验证 JWT，并确保只能查自己的收藏夹
    current_user = get_current_user(authorization)
    if current_user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="无权访问他人的收藏夹")

    conn = sqlite3.connect("music_hash.db")
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

@app.post("/search")
async def api_search_music(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None)   # 可选，不强制要求登录
):
    """
    接收前端上传的音频文件，进行听歌识曲。
    已登录用户的搜索记录会自动写入历史表。
    """
    # 1. 将用户上传的音频临时存到本地
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. 调用核心搜索逻辑，默认返回最相似的 5 首歌
        results = search_music(temp_file_path, INDEX, MODEL)

        # 🌟 已登录用户：静默记录搜索历史（失败不影响检索结果返回）
        if authorization and authorization.startswith("Bearer "):
            try:
                user = get_current_user(authorization)
                top = results[0] if results else None
                conn = sqlite3.connect("music_hash.db")
                conn.execute(
                    "INSERT INTO search_history (user_id, query_filename, top_title, top_artist, top_audio_url, top_file_path) VALUES (?,?,?,?,?,?)",
                    (
                        user["user_id"],
                        file.filename,
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
async def get_history(authorization: Optional[str] = Header(None)):
    """获取当前用户最近 50 条搜索历史，按时间倒序"""
    user = get_current_user(authorization)
    conn = sqlite3.connect("music_hash.db")
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
async def clear_history(authorization: Optional[str] = Header(None)):
    """清空当前用户的全部搜索历史"""
    user = get_current_user(authorization)
    conn = sqlite3.connect("music_hash.db")
    conn.execute("DELETE FROM search_history WHERE user_id=?", (user["user_id"],))
    conn.commit()
    conn.close()
    return {"code": 200, "message": "搜索历史已清空"}


if __name__ == "__main__":
    # 启动服务器，运行在本地 8000 端口
    uvicorn.run(app, host="127.0.0.1", port=8000)