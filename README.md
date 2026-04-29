# VECTORTUNE

基于深度哈希网络的音乐音频相似度检索系统 / 版权查重平台

## 简介

VECTORTUNE 是一个面向纯净数字音乐与带白噪环境的音频检索平台。系统采用 CNN 提取 128 维二进制哈希指纹（结合 Triplet Loss 与动态中值哈希解决特征坍塌），利用 Faiss `IndexBinaryFlat` 对十万级子指纹做毫秒级汉明距离匹配。

目前系统对数字原曲实现 **0 bit 完美命中**、对白噪音具备抵抗性。

## 核心特性

- **深度哈希指纹**：基于 Triplet Loss 训练的 CNN 网络，将音频 Mel 频谱压缩为 128 位二进制哈希
- **毫秒级检索**：Faiss 二进制向量索引，汉明距离快速匹配
- **抗噪能力**：在线白噪声数据增强，对白噪音环境具备鲁棒性
- **JWT 无状态鉴权**：7 天有效期 JWT，支持 httpOnly Cookie 和 Bearer Token
- **单文件前端**：Vue 3 SPA + WaveSurfer.js 波形可视化 + Paper Shaders 动态背景

## 技术栈

- **后端**：FastAPI + Uvicorn + SQLite
- **算法**：PyTorch + Librosa + Faiss-cpu
- **前端**：Vue 3 (CDN) + WaveSurfer.js + Paper Shaders (WebGPU)
- **部署**：Docker + Docker Compose + Caddy (自动 HTTPS)

## 项目结构

```
music-search-system/
├── main.py              # FastAPI 入口，路由中枢
├── search.py            # 在线检索核心逻辑
├── model.py             # AudioHashNet 神经网络定义
├── dataset.py           # Triplet 数据集构造
├── train.py             # 模型训练脚本
├── batch_extract.py     # 离线建库脚本
├── precompute_cache.py  # 波形缓存预计算
├── make_noise.py        # 噪声增强脚本
├── index.html           # 单文件前端 SPA
├── config.py            # 环境变量配置
├── requirements.txt     # Python 依赖
├── Dockerfile           # 容器镜像定义
├── docker-compose.yml   # 服务编排
├── Caddyfile            # Caddy 反向代理配置
├── .env.example         # 环境变量模板
├── data/                # 数据存储中心
│   ├── raw/             # 原始音频（不入库）
│   └── waveform_cache/  # 波形缓存
├── checkpoints/         # 模型权重
└── tests/               # 单元测试
```

## 快速开始

### 1. 环境配置

复制环境变量模板并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，设置必填项：

```bash
# 生成 JWT 密钥
JWT_SECRET=$(openssl rand -hex 32)

# 本地调试时设为 0 或不设置
COOKIE_SECURE=0
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行服务

```bash
export JWT_SECRET="$(openssl rand -hex 32)"
python main.py
```

访问 `http://127.0.0.1:8000/` 即可使用。

## 开发指南

### 构建音频索引

扫描 `data/raw/` 下全量音频，生成 Faiss 索引：

```bash
python batch_extract.py
```

### 训练模型

训练 CNN 深度哈希模型：

```bash
python train.py
```

### 预计算波形缓存

加速训练时的音频解码：

```bash
python precompute_cache.py
```

### 生成噪声测试音频

```bash
python make_noise.py
```

### 运行测试

```bash
pytest -v tests/
```

## 生产部署

系统已容器化，使用 Docker Compose 一键部署：

```bash
docker compose up -d --build
```

**部署要求**：
- 宿主机预先放置 `music_hash.db` 与 `music_hash.index`
- 配置 `.env` 中的 `JWT_SECRET` 和 `COOKIE_SECURE=1`
- 确保域名已解析到服务器（当前部署域名：`evanesce.site`）

Caddy 会自动申请并续期 Let's Encrypt SSL 证书。

## 环境变量说明

| 变量 | 必填 | 说明 |
|------|------|------|
| `JWT_SECRET` | 是 | JWT 签名密钥，使用 `openssl rand -hex 32` 生成 |
| `COOKIE_SECURE` | 否 | 生产 HTTPS 设为 `1`，本地调试设为 `0` 或不设置 |
| `CORS_ORIGINS` | 否 | CORS 白名单，逗号分隔 |
| `MAX_UPLOAD_SIZE` | 否 | 最大上传大小（字节），默认 52428800 |
| `RATE_LIMIT_SEARCH` | 否 | 搜索接口限流（请求/分钟），默认 30 |
| `RATE_LIMIT_AUTH` | 否 | 认证接口限流，默认 100 |
| `RATE_LIMIT_GENERAL` | 否 | 通用接口限流，默认 300 |

## 注意事项

- **绝对禁止**将 `.env` 文件提交到 Git（已在 `.gitignore` 中忽略）
- **绝对禁止**修改二值化策略，必须维持 `binary_code = (hash_out > hash_out.median())` 的动态中值判定
- 生产环境必须通过环境变量注入 `JWT_SECRET`，代码无默认值
- `wheels/` 目录（337 MB 离线依赖包）不入库

## 许可证

本项目仅供学习与研究使用。

## 在线演示

生产环境：https://evanesce.site
