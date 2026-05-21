import os

# 本地开发时自动从项目根目录的 .env 加载环境变量。
# 生产 Docker 通过 docker-compose.yml 的 env_file 注入，dotenv 不会覆盖已有值（override=False）。
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass


DB_PATH = os.environ.get("DB_PATH", "music_hash.db")
INDEX_PATH = os.environ.get("INDEX_PATH", "music_hash.index")
MODEL_PATH = os.environ.get("MODEL_PATH", "./checkpoints/hash_model_v2_epoch_10_best.pth")
RAW_PATH = os.environ.get("RAW_PATH", "./data/raw")
TOP_K = int(os.environ.get("TOP_K", 150))
TOP_N_RESULTS = int(os.environ.get("TOP_N_RESULTS", 5))

SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", 22050))
WINDOW_DURATION = float(os.environ.get("WINDOW_DURATION", 5.0))
N_MELS = int(os.environ.get("N_MELS", 128))

JWT_SECRET = os.environ.get("JWT_SECRET")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_DAYS = int(os.environ.get("JWT_EXPIRE_DAYS", 7))

MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 50 * 1024 * 1024))
ALLOWED_AUDIO_EXTENSIONS = {
    ext.strip().lower()
    for ext in os.environ.get("ALLOWED_AUDIO_EXTENSIONS", ".mp3,.wav,.flac,.ogg,.m4a,.aac").split(",")
    if ext.strip()
}

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
CORS_ORIGIN_LIST = [origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()]

RATE_LIMIT_SEARCH = int(os.environ.get("RATE_LIMIT_SEARCH", 30))
RATE_LIMIT_AUTH = int(os.environ.get("RATE_LIMIT_AUTH", 100))
RATE_LIMIT_GENERAL = int(os.environ.get("RATE_LIMIT_GENERAL", 300))
