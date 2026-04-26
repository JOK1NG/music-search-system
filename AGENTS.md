# AI 音乐相似度检索系统 - Agent 指南 (AGENTS.md)

## Project Overview

纯净及带一定白噪环境下的音乐音频检索与版权查重平台。采用 CNN 提取 128维哈希特征 (结合 Triplet Loss 与 Median Hashing 解决特征坍塌)，利用 Faiss 进行毫秒级向量匹配。前端为轻量级 Vue3 单文件 (CDN 引入)。系统目前对白噪音具备抵抗性，但对复杂外录环境（域偏移）缺乏抗干扰能力。
系统由四个核心驱动：

1. **AI 算法与信号处理**: `PyTorch` (CNN) + `Librosa` (DSP)，利用 Triplet Loss 提取高维声学特征并二值化。
2. **极速向量检索**: `Faiss` 引擎，基于二进制哈希码计算汉明距离。
3. **Web 后端**: `FastAPI` + `SQLite`，处理高并发检索、用户鉴权（`bcrypt`）。
4. **前端交互**: 单文件 Vue 3 SPA 配合 `WaveSurfer.js`。

## Local Development Commands

本项目是一个极其轻量的全栈系统，无需复杂的包管理（如 npm 或复杂的依赖环境，只需核心 Python 库）。

### 1. 运行核心服务

启动 FastAPI 后端（也是对外的 API 及静态文件服务器）。

```bash
python main.py
```

前端页面入口：直接在浏览器中打开根目录下的 `index.html` 即可，零插件起步。

### 2. 数据库与离线建库 (必要时运行)

初始化 SQLite 数据库 (`music_hash.db`)，包含用户表与收藏表：

```bash
python init_db.py
```

提取 `data/raw/` 下所有音频，执行特征压缩并生成离线索引 (`music_hash.index` 和 SQLite 映射)：

```bash
python batch_extract.py
```

### 3. 模型训练与造数据 (算法迭代运行)

训练 CNN 深度哈希模型（生成模型权重至 `checkpoints/`），默认已调优超参数 `lr=1e-5`, `margin=5.0`：

```bash
python train.py
```

快速构造带背景噪音的测试音频（用于验证模型抗噪能力）：

```bash
python make_noise.py
```

## Architecture / Directory Responsibilities

目录遵循扁平化微服务结构，核心责任划分明确：

- `./main.py`: **系统入口**。 API 路由控制中心（/search, /login, /register），配置跨域及挂载 data/raw 静态目录，负责 JWT/Session 鉴权。
- `./search.py`: **核心业务大脑**。在线检索核心大脑。处理前端上传音频（Librosa去静音/归一化），执行模型推理，调用 Faiss 返回多候选，合并比对结果。
- `./model.py`: **AI 神经网络定义**。定义深层 AudioHashNet 架构，负责音频频域到 128 位哈希映射的神经网络主体。
- `./dataset.py`: **数据工厂**。构造 Triplet 数据流，执行在线数据增强（添加白噪）。
- `./train.py`: **炼丹炉**。训练守护进程。基于 Triplet Loss 控制模型收敛。
- `./batch_extract.py`: 离线建库脚本。基于“滑动窗口策略”处理全量库，刷新 Faiss 二进制索引和 SQLite 的 songs 表。
- `./index.html`: **单文件前端**。Vue3 组合式 API + WaveSurfer.js 前端应用，单文件全栈结构。
- `./data/`: **存储中心**。存放原始音频 `raw/` 目录以及数据库文件。
- `./checkpoints/`: **模型权重库**。存放 `.pth` 模型权重供推理使用。

## Coding Rules & Safety Rails (Agent 注意事项)

What NOT to Do (Safety Rails)

1.绝对禁止修改密码加密方式：
   不要 试图引入旧版 passlib，直接使用原生 bcrypt 库处理密码加盐。因为旧库有检测机制引发的 72 Bytes 长度超限崩溃 Bug。
2.绝对禁止打破预处理机制“非对称性”：
   不要 使得 search.py (在线查询) 和 batch_extract.py (库构建) 两者的特征截断、特征二值化策略（必须维持动态中值哈希判定 hash_out.median()）出现丝毫偏差，否则会发生毁灭性的哈希空间坍塌与 Domain Shift。
3.绝对禁止前端“过度工程化”：
   不要 创建 Vue CLI、Vite 或 package.json。本项目严格维持单文件开发，通过 `<script src="https://unpkg.com/...">` 的无包管理 CDN 模式引入一切环境。
4.禁止随意破坏或重置数据库结构：
   业务关系已依赖 users 和 favorites 表的主键和 UNIQUE 逻辑。不要 随手运行 init_db.py 内的 DROP 逻辑，会清空整个产品的本地用户数据。

## Before changing code, read these files first

- `background.md` (理解项目技术背景、已攻克的难点与算法边界)
- `main.py` (了解当前系统提供的 API 能力和路由限制)
- `search.py` (查看目前音频预处理和检索流程的具体参数设定)
- `search.py` 和 `batch_extract.py` (必须同步阅读，了解系统滑动切片的核心搜索特征流)。
