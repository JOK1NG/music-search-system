# 代码审查报告：音乐音频相似度检索系统

> 审查日期：2026-04-24 | 审查范围：全项目 12 个 Python 文件 + 前端 + Docker 配置

---

## 一、严重 - P0（安全 / 稳定性威胁）

### 1. 临时文件路径穿越漏洞

- **文件**: `main.py:280`
- **问题**: `f"temp_{file.filename}"` 将用户提供的文件名直接拼入本地路径，攻击者可构造 `../../etc/passwd` 实现任意文件读写
- **修复**: 用 `uuid4().hex + os.path.splitext(file.filename)[1]` 生成不可预测的安全文件名

### 2. JWT 密钥硬编码 + docker-compose.yml 泄露

- **文件**: `main.py:27`、`docker-compose.yml:19`
- **问题**:
    - `main.py` 中 `JWT_SECRET` 回退值为 `"vectortune-dev-secret-change-in-production"`，未设环境变量时使用弱密钥
    - `docker-compose.yml` 明文写的 `JWT_SECRET=TencentCloud_Production_Secret_ChangeMe` 被纳入版本控制
- **修复**: 启动时若未从环境变量读取 `JWT_SECRET` 则直接拒绝启动（移除回退值）；将密钥移入 `.env` 并加入 `.gitignore`

### 3. CORS 全开 + credentials 组合不安全

- **文件**: `main.py:122`
- **问题**: `allow_origins=["*"]` 配合 `allow_credentials=True` 是不安全配置，任意来源可带凭据请求
- **修复**: 生产环境改为白名单域名列表

### 4. 文件上传无大小与类型限制

- **文件**: `main.py:271-274`
- **问题**: `/search` 端点未检查 `Content-Length`，也未校验文件 MIME 类型，大文件可耗尽磁盘/内存
- **修复**: 添加 `python-multipart` 的大小限制，校验文件魔数

### 5. 无速率限制

- **文件**: `main.py` 全部 API 端点
- **问题**: 所有端点（尤其 `/search` 涉及 GPU/CPU 推理）无限流，单 IP 可打垮服务
- **修复**: 引入 `slowapi` 或中间件令牌桶限流

---

## 二、高 - P1（架构 / 可靠性缺陷）

### 6. 核心预处理逻辑重复 —— 违反非对称性原则

- **文件**: `search.py:42-83` vs `batch_extract.py:67-97`
- **问题**: mel 频谱提取、短音频 padding、中值哈希二值化逻辑几乎完全重复。AGENTS.md 明确规定两处特征截断与二值化策略"绝对禁止出现丝毫偏差"，分头维护极易引发哈希空间坍塌
- **修复**: 抽取公共模块 `preprocessing.py`，统一 `build_mel_spectrogram()`、`binarize_hash()` 函数，`search.py` 和 `batch_extract.py` 共用

### 7. 配置散落、魔法值硬编码

- **文件**: `search.py:10-12`、`batch_extract.py:11-14`、`dataset.py`、`train.py`
- **问题**: `MODEL_PATH`、`DB_PATH`、`INDEX_PATH`、采样率 `22050`、窗口长度 `5.0`、top_k `150`、时长上限 `15.0`/`30.0` 等在多个文件重复定义，共 10+ 处魔法值
- **修复**: 创建 `config.py` 集中所有常量，通过环境变量覆写

### 8. 全项目使用 `print()` 替代正式日志

- **文件**: 所有 `*.py`
- **问题**: 调试与生产日志不可分离、无级别过滤、无法写入持久化存储
- **修复**: 引入标准库 `logging`，为每个模块配置独立 logger，用 `UVICORN_LOG_LEVEL` 控制

### 9. 每次请求创建/销毁 SQLite 连接

- **文件**: `main.py` 中 `register`、`login`、`toggle_favorite`、`get_favorites`、`search`、`get_history`、`clear_history`
- **问题**: 每个端点独立调用 `sqlite3.connect()` / `close()`，高并发下频繁建连性能极差且易锁冲突
- **修复**: 使用 `contextlib.contextmanager` 统一连接管理，或开启 WAL 模式 + 连接池

### 10. 静默吞异常，无迹可查

- **文件**: `batch_extract.py:103-104`、`main.py:307-308`
- **问题**:
    - `batch_extract.py`: `except Exception: pass` —— 建库时遇到坏文件无任何日志
    - `main.py`: 搜索历史写入失败被 `try/except: pass` 彻底吞掉
- **修复**: 至少记录日志；`batch_extract.py` 收集失败文件列表并在结束时统一报告

### 11. 缺少健康检查端点

- **文件**: `main.py`
- **问题**: docker-compose 未配置 `healthcheck`，`depends_on` 仅保证容器启动顺序而非服务就绪状态；Caddy 可能在 FastAPI 未加载完模型时就开始代理
- **修复**: 添加 `GET /health` 返回 `{"status": "ok", "index_loaded": bool, "model_loaded": bool}`

### 12. 缺少优雅关闭逻辑

- **文件**: `main.py:363-365`
- **问题**: `uvicorn.run` 无 lifespan 处理器，不存在 DB 关闭、Faiss 索引释放、临时文件清理机制
- **修复**: 实现 FastAPI lifespan context manager，在 `shutdown` 中关闭 DB 连接、清理临时文件

---

## 三、中 - P2（代码质量 / 性能）

### 13. 零自动化测试覆盖

- **文件**: 全项目
- **问题**: 无任何测试框架或断言类测试用例；`utils/search_test.py` 仅手动验证脚本
- **修复**: 引入 `pytest`，优先覆盖搜索管线 (`search.py`)、身份验证 (`main.py` 鉴权端点)、数据库 CRUD

### 14. search_music 内 N×M 查询问题

- **文件**: `search.py:95-104`
- **问题**: 嵌套循环中逐条 `SELECT ... WHERE faiss_id=?`，最坏 3 窗口 × 150 候选 = 450 次 DB 查询
- **修复**: 先收集所有 faiss_id，用 `WHERE faiss_id IN (...)` 一条 SQL 批量取回

### 15. batch_extract 逐窗口推理未批量化

- **文件**: `batch_extract.py:75-97`
- **问题**: 每首歌的每个滑动窗口独立调 `model(tensor)`，无法利用 GPU 批处理能力
- **修复**: 堆叠同曲所有窗口 tensor 一次性推入模型，GPU 利用率可提升数倍

### 16. search_music 中频谱重复计算

- **文件**: `search.py:71-72`
- **问题**: 每个 5 秒窗口分别执行 `melspectrogram` + `power_to_db`（大量 FFT），而 15 秒音频的大频谱可一次计算后切片
- **修复**: 先对完整 15 秒提取大 mel 频谱，再按时间轴索引切片，FFT 次数减少 66%

### 17. 依赖未锁定版本

- **文件**: `requirements.txt`
- **问题**: 所有包无版本号，不同时间 pip install 可能拉取不兼容新版
- **修复**: 使用 `pip freeze` 生成精确版本约束，或至少写 `fastapi>=0.100,<1.0` 区间

### 18. Dockerfile COPY . . 存在膨胀风险

- **文件**: `Dockerfile:21`
- **问题**: 完全依赖 `.dockerignore` 排除 `.venv/`，一旦配置失误数百 MB 虚拟环境被打进镜像
- **修复**: 改为显式 `COPY main.py search.py ...` 白名单模式，或加固 `.dockerignore`

### 19. 数据库缺少外键约束与索引

- **文件**: `main.py:92-113`
- **问题**:
    - `favorites.user_id`、`search_history.user_id` 无外键，无法保证引用完整性
    - `search_history` 表按 `id DESC` 排序取最近 50 条，但 `searched_at` 无索引
- **修复**: 启用 `PRAGMA foreign_keys = ON`，添加 `users`、`searched_at` 索引

### 20. JWT 令牌存储于 localStorage 存在 XSS 风险

- **文件**: `index.html:1384-1385`
- **问题**: `localStorage.setItem('music_app_token', ...)` —— 任何 XSS 注入可直接偷走令牌
- **修复**: 改用 `httpOnly` + `Secure` Cookie 存储 JWT，配合 `SameSite=Strict`

---

## 四、低 - P3（增强 / 锦上添花）

| # | 问题 | 位置 |
|---|------|------|
| 21 | 缺少 CI/CD 流水线（GitHub Actions / 自建 Runner） | — |
| 22 | 无可观测性指标（无 Prometheus metrics、无 OpenTelemetry 追踪） | — |
| 23 | `train.py` 无早停 (early stopping)、无 checkpoint 恢复、无 TensorBoard | `train.py` |
| 24 | `init_db.py` 仍操作旧的 `data/music_meta.db`，与主库 `music_hash.db` 分裂，易造成维护混淆 | `init_db.py` |
| 25 | `make_noise.py` 硬编码源文件路径 `./data/raw/000/000002.mp3` | `make_noise.py:7` |
| 26 | WaveSurfer 实例在 Vue 组件卸载时未调用 `destroy()`，存在内存泄漏 | `index.html` |
| 27 | 前端无 CSP (Content-Security-Policy) 响应头防御 XSS | `main.py` |
| 28 | `model.py` Dropout(0.5) 偏高，可能在学习率极低时欠拟合 | `model.py:34` |
| 29 | 无 VAD (Voice Activity Detection)，前置/后置静音区浪费滑动窗口位 | `search.py` |
| 30 | `requirements.txt` 缺少 `python-dotenv`（配合 .env 方案） | `requirements.txt` |

---

## 五、执行路线图

```
v1.1 — 安全加固（P0 + P1）
  ├─ P0-1  修复路径穿越
  ├─ P0-2  移除 JWT 硬编码密钥
  ├─ P0-3  限制 CORS 来源
  ├─ P0-4  文件上传大小/类型校验
  ├─ P0-5  接口速率限制
  ├─ P1-6  统一预处理公共模块
  ├─ P1-7  集中配置管理
  ├─ P1-8  引入 logging 模块
  ├─ P1-9  SQLite 连接池化
  ├─ P1-10 消除静默异常吞没
  ├─ P1-11 添加 /health 端点
  └─ P1-12 实现优雅关闭

v1.2 — 质量提升（P2）
  ├─ P2-13 添加测试框架 + 核心管线测试
  ├─ P2-14 批量 SQL 查询优化
  ├─ P2-15 batch_extract 推理批量化
  ├─ P2-16 频谱计算去重
  ├─ P2-17 锁定依赖版本
  ├─ P2-18 Dockerfile 白名单 COPY
  ├─ P2-19 数据库外键 + 索引
  └─ P2-20 JWT 迁移至 httpOnly Cookie

v1.3 — backlog（P3）
  └─ P3-21~30 按需消化
```
