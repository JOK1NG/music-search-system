# AI 音乐相似度检索系统 - Agent 指南 (AGENTS.md)

## Project Overview

**VECTORTUNE**：面向纯净数字音乐与带一定白噪环境的音频检索 / 版权查重平台。
系统采用 CNN 提取 128 维二进制哈希指纹（结合 Triplet Loss 与动态中值哈希解决特征坍塌），利用 Faiss `IndexBinaryFlat` 对十万级子指纹做毫秒级汉明距离匹配。
目前系统对数字原曲实现 **0 bit 完美命中**、对白噪音具备抵抗性，但对复杂外录环境（设备底噪、房间混响的物理域偏移）仍缺乏抗干扰能力，该方向已纳入未来研究路线。

系统已完成全栈工程化并部署至生产环境 `https://evanesce.site`（腾讯云 Ubuntu ECS + Docker + Caddy 自动 HTTPS），由四个核心层驱动：

1. **AI 算法与信号处理层**：`PyTorch`（CNN 深度哈希）+ `Librosa`（音频 DSP），利用 Triplet Loss 学习高维声学特征并动态二值化。
2. **极速向量检索层**：`Faiss` 引擎，基于二进制哈希码计算汉明距离。
3. **全栈 Web 后端**：`FastAPI` + `Uvicorn` + `SQLite`，支持 `bcrypt` 密码哈希与 `JWT` 无状态鉴权。
4. **前端交互层**：单文件 Vue 3 SPA（CDN 引入，无构建步骤）+ `WaveSurfer.js` 波形可视化 + `Paper Shaders` (WebGPU) 品牌级背景。

## Local Development Commands

本项目是一个极其轻量的全栈系统，无复杂包管理（零 npm / 零前端构建工具，只需核心 Python 库）。

### 1. 运行核心服务

启动 FastAPI 后端（同时担任对外的 API 服务器与 `index.html` 静态分发入口）：

```bash
python main.py
```

前端页面入口：直接在浏览器中打开 `http://127.0.0.1:8000/` 即可；文件级打开 `index.html` 也能用，但推荐经 FastAPI 访问以便 `window.location.origin` 正确推断 API 地址。

### 2. 数据库与离线建库 (必要时运行)

初始化 SQLite 数据库 (`music_hash.db`)，生成 `users`、`favorites`、`search_history`、`songs` 四张业务表：

```bash
python init_db.py
```

扫描 `data/raw/` 下全量音频，执行滑动窗口切片 + 特征压缩，生成离线索引 (`music_hash.index`) 与 `songs` 元数据映射：

```bash
python batch_extract.py
```

预计算音频波形缓存（将 MP3/WAV 解码为 NumPy 数组存至 `data/waveform_cache/`，大幅加速训练时 `librosa.load` 解码）：

```bash
python precompute_cache.py
```

### 3. 模型训练与造数据 (算法迭代运行)

训练 CNN 深度哈希模型（生成权重至 `checkpoints/`），默认调优超参数 `lr=1e-5`, `margin=5.0`：

```bash
python train.py
```

快速构造带背景噪音的测试音频（用于验证模型抗噪能力）：

```bash
python make_noise.py
```

### 4. 单元测试

本项目具备 `pytest` 单元测试覆盖（鉴权、JWT 签发/校验、音频预处理、速率限制）：

```bash
pytest -v tests/
```

### 5. 生产部署（Docker Compose + Caddy）

系统已容器化，单机一行命令起所有服务（FastAPI + Caddy 反代 + 自动申请 HTTPS 证书）：

```bash
docker compose up -d --build
```

Caddy 监听 80/443，反代到 `music-search:8000`。宿主机必须预先放置 `music_hash.db` 与 `music_hash.index`（通过 volume 挂载持久化）。

## Architecture / Directory Responsibilities

目录遵循扁平化微服务结构，核心责任划分明确：

- `./main.py`：**系统入口**。FastAPI 路由中枢（`/`、`/register`、`/login`、`/search`、`/favorite/toggle`、`/favorites/{uid}`、`/history` GET/DELETE），承担 JWT 签发/校验、CORS 配置、`/audio` 静态挂载。
- `./search.py`：**在线检索大脑**。处理前端上传音频，执行 RMS 能量选秀 + 滑动窗口切片，调用模型推理，经 Faiss 返回多候选并合并比对。
- `./model.py`：**AI 神经网络定义**。`AudioHashNet` 架构，负责 Mel 频谱到 128 位哈希的映射。
- `./dataset.py`：**数据工厂**。构造 Triplet `(anchor, positive, negative)` 数据流，执行在线白噪声数据增强。
- `./train.py`：**炼丹炉**。基于 Triplet Margin Loss 控制模型收敛。
- `./batch_extract.py`：**离线建库脚本**。滑动窗口扫全量音频库，刷新 Faiss 二进制索引 + `songs` 表。
- `./precompute_cache.py`：**训练加速缓存**。提前解码 MP3/WAV 为 `.npy`，跳过训练时最昂贵的音频解码步骤。
- `./make_noise.py`：**噪声增强脚本**。合成带白噪音版本的测试音频。
- `./index.html`：**单文件前端**。Vue 3 组合式 API + `WaveSurfer.js` + `Paper Shaders` (WebGPU) 背景的单文件 SPA。
- `./favicon.svg`、`./vectortun.svg`：VECTORTUNE 品牌 Logo 与 favicon。
- `./data/`：**存储中心**。原始音频 `raw/`、波形缓存 `waveform_cache/`、元数据 `music_meta.db` 等。
- `./checkpoints/`：**模型权重库**。存 `.pth` 权重供推理加载。
- `./tests/`：**测试套件**。含 `conftest.py`、`test_auth.py`、`test_jwt.py`、`test_preprocessing.py`。
- `./Dockerfile`：容器镜像定义。基于 `python:3.11-slim`，安装 `ffmpeg` + `libsndfile1`，离线从 `wheels/` 装 Python 依赖。
- `./docker-compose.yml`：双服务编排（`music-search` FastAPI 容器 + `caddy_proxy` Caddy 容器）。
- `./Caddyfile`：Caddy 反向代理配置，自动为 `evanesce.site` 申请 / 续期 Let's Encrypt 证书。
- `./wheels/`：离线 pip 依赖包（`.gitignore` 屏蔽，不入仓库；337 MB，由 `pip download` 生成）。
- `./requirements.txt`：Python 依赖清单（`fastapi`, `uvicorn`, `pydantic`, `bcrypt`, `PyJWT`, `librosa`, `torch`, `numpy`, `faiss-cpu`, `tinytag`, `soundfile`, `python-multipart`）。
- `./background.md`：项目技术背景白皮书（算法踩坑史、架构决策）。
- `./现有项目的总结.md`：里程碑回顾文档（面向毕业论文 / README 的完整叙事）。

## Coding Rules & Safety Rails (Agent 注意事项)

### What NOT to Do

1. **绝对禁止修改密码加密方式**
   不要 试图引入旧版 `passlib`。本项目直接用原生 `bcrypt.hashpw` / `bcrypt.checkpw`。因为老版 `passlib` 会发送超长测试字符串触发新版 `bcrypt` 的 72 bytes 超限保护，导致 ASGI 服务器崩溃。

2. **绝对禁止打破预处理机制"非对称性"**
   不要 使得 `search.py`（在线查询）和 `batch_extract.py`（库构建）的特征截断、二值化策略出现丝毫偏差。**二值化必须统一维持 `binary_code = (hash_out > hash_out.median())` 的动态中值判定**。任何改动（比如退回 `> 0` 死阈值）会瞬间引发哈希空间坍塌 + Domain Shift，全库命中率崩盘。

3. **绝对禁止前端"过度工程化"**
   不要 创建 Vue CLI / Vite / webpack / `package.json` / `node_modules/`。本项目**严格维持单文件 `index.html` 开发**，所有依赖通过 `<script src="https://unpkg.com/...">` 或 `<script type="module" src="https://esm.sh/...">` 的 CDN 方式无包管理引入。

4. **绝对禁止随意 DROP 或重置数据库结构**
   业务已深度依赖 `users.id` 主键、`favorites (user_id, song_path) UNIQUE`、`search_history` 时间戳等约束。不要 手动触发 `init_db.py` 内的 DROP 逻辑，**会清空整个产品的线上用户数据**。

5. **绝对禁止把 `wheels/` 提交进 git**
   `wheels/` 体积 337 MB，已在 `.gitignore` 中屏蔽。生产部署时通过 `pip download -r requirements.txt -d wheels/` 本地生成，走宿主机挂载或镜像构建上下文。一旦入库会立即撑爆仓库。

6. **绝对禁止硬编码 `JWT_SECRET`**
   生产环境必须通过环境变量 `JWT_SECRET` 注入强随机密钥（当前 `docker-compose.yml` 通过 `environment` 字段读取）。代码里的默认值 `"vectortune-dev-secret-change-in-production"` 只在本地开发时生效。一旦泄漏，任何人都能伪造任意用户的 token。

7. **修改 `main.py` 路由时务必保留 JWT 鉴权顺序**
   所有涉及用户数据的端点（收藏、历史）都走 `get_current_user(authorization)` 从 JWT payload 提取 `user_id`，**不要信任前端传入的 `user_id` 字段**。绕过此机制会引入水平越权漏洞。

8. **生产部署更新 UI 必须走 `docker compose up -d --build`**
   单改 `index.html` 不触发容器重建是无效的（`Dockerfile` 用 `COPY . .` 把前端烧进镜像）。如需前端热更，应将 `index.html` 加入 `docker-compose.yml` 的 volume 挂载段。

## Production Deployment

| 项 | 配置 |
|---|---|
| **域名** | `evanesce.site` + `www.evanesce.site` |
| **宿主机** | 腾讯云轻量 ECS，Ubuntu 22.04，2 vCPU / 3.6 GB RAM / 59 GB SSD |
| **项目目录** | `/root/music-search-system/`（⚠️ 不是 `/home/ubuntu/`） |
| **反代链路** | 用户 → Caddy(443, HTTPS) → `music-search:8000` (FastAPI) |
| **SSL 证书** | Caddy 自动申请 / 续期 Let's Encrypt，持久化在 `caddy_data` volume |
| **数据 volumes** | `./data`、`./checkpoints`、`./music_hash.db`、`./music_hash.index` 全部宿主机挂载 |
| **更新工作流** | `sudo -i` → `cd /root/music-search-system` → `git pull origin <branch>` → `docker compose up -d --build music-search` |
| **不重建镜像即可热更场景** | 修改 `Caddyfile` / `docker-compose.yml` / 纯文档 |

## Before changing code, read these files first

- `background.md`（理解项目技术背景、已攻克的难点与算法边界）
- `现有项目的总结.md`（掌握全栈里程碑与架构决策全貌）
- `main.py`（了解当前系统提供的 API 能力、JWT 鉴权闭环、路由限制）
- `search.py`（查看音频预处理 + 滑动窗口检索流程的具体参数设定）
- `search.py` 和 `batch_extract.py` **必须同步阅读**（了解系统滑动切片的特征流，严守对称性）
- 改路由 / 鉴权前同时读 `tests/test_auth.py` 与 `tests/test_jwt.py`，避免破坏既有测试契约
