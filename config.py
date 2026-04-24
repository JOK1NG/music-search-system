import os


# --- 路径 ---
DB_PATH = os.environ.get("DB_PATH", "music_hash.db")
INDEX_PATH = os.environ.get("INDEX_PATH", "music_hash.index")
MODEL_PATH = os.environ.get("MODEL_PATH", "./checkpoints/hash_model_epoch_3.pth")
RAW_PATH = os.environ.get("RAW_PATH", "./data/raw")
WAVEFORM_CACHE_PATH = os.environ.get("WAVEFORM_CACHE_PATH", "./data/waveform_cache")

# --- 音频参数 ---
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", 22050))
WINDOW_DURATION = float(os.environ.get("WINDOW_DURATION", 5.0))
MAX_DURATION_SEARCH = float(os.environ.get("MAX_DURATION_SEARCH", 15.0))
MAX_DURATION_BUILD = float(os.environ.get("MAX_DURATION_BUILD", 30.0))
N_WINDOWS_SEARCH = int(os.environ.get("N_WINDOWS_SEARCH", 3))
N_MELS = int(os.environ.get("N_MELS", 128))
HASH_LENGTH = int(os.environ.get("HASH_LENGTH", 128))

# --- 检索 ---
TOP_K = int(os.environ.get("TOP_K", 150))
TOP_N_RESULTS = int(os.environ.get("TOP_N_RESULTS", 5))

# --- JWT ---
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET 环境变量未设置，请在生产环境中配置强随机密钥。"
        "例如: export JWT_SECRET=$(openssl rand -hex 32)"
    )
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_DAYS = int(os.environ.get("JWT_EXPIRE_DAYS", 7))

# --- CORS ---
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
CORS_ORIGIN_LIST = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]

# --- 速率限制 ---
RATE_LIMIT_SEARCH = int(os.environ.get("RATE_LIMIT_SEARCH", 30))      # /search 每分钟请求数
RATE_LIMIT_AUTH = int(os.environ.get("RATE_LIMIT_AUTH", 10))          # 登录/注册 每分钟请求数
RATE_LIMIT_GENERAL = int(os.environ.get("RATE_LIMIT_GENERAL", 60))    # 其他端点 每分钟请求数

# --- 上传限制 ---
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 50 * 1024 * 1024))  # 50MB
ALLOWED_AUDIO_EXTENSIONS = os.environ.get(
    "ALLOWED_AUDIO_EXTENSIONS", ".mp3,.wav,.flac,.ogg,.m4a,.aac"
).split(",")

# --- 训练 ---
TRAIN_LR = float(os.environ.get("TRAIN_LR", 1e-5))
TRAIN_MARGIN = float(os.environ.get("TRAIN_MARGIN", 5.0))
TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 3))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 64))
TRAIN_NUM_WORKERS = int(os.environ.get("TRAIN_NUM_WORKERS", 4))
TRAIN_NOISE_FACTOR = float(os.environ.get("TRAIN_NOISE_FACTOR", 0.01))
TRAIN_DROPOUT = float(os.environ.get("TRAIN_DROPOUT", 0.3))
TRAIN_PATIENCE = int(os.environ.get("TRAIN_PATIENCE", 5))
TRAIN_MIN_DELTA = float(os.environ.get("TRAIN_MIN_DELTA", 1e-4))
TRAIN_CHECKPOINT_DIR = os.environ.get("TRAIN_CHECKPOINT_DIR", "./checkpoints")
TRAIN_RESUME_CHECKPOINT = os.environ.get("TRAIN_RESUME_CHECKPOINT", "latest.ckpt")
TRAIN_LOG_DIR = os.environ.get("TRAIN_LOG_DIR", "./runs")
