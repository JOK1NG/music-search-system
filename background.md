# 项目背景总结：VECTORTUNE 音乐音频相似度检索系统 / 海量数字音乐版权查重与相似度检索平台

## 项目概述

本项目旨在设计并实现一个端到端的"音乐音频相似度检索系统"，能够支撑海量数字音乐的极速检索与原曲识别。系统不仅能对纯净数字音频实现 **0 bit 完美精准命中**，同时具备对白噪声等环境干扰的鲁棒性。

随着工程与算法的双线演进，项目从最初"能在实验室跑通的算法 Demo"，依次跃升为：**完整 Web 产品 → 具备 JWT 鉴权的多用户 SaaS 原型 → 容器化部署 + HTTPS 自动续期的上线态全栈应用**，目前已以 **VECTORTUNE** 品牌正式运行在 `https://evanesce.site`（腾讯云 Ubuntu ECS），对外提供音频上传、原曲识别、音频试听、个人收藏与识别历史回溯等完整体验。

## 核心技术架构

项目采用现代化、轻量级的前后端分离架构，打通了"四核驱动"的数据链路：

- **AI 算法与信号处理层**：基于 Python 3.11。`Librosa` 执行音频数字信号处理（DSP）：去静音、响度归一化、Mel 频谱图提取；`PyTorch` 部署自研的深度哈希网络（DHN）——`AudioHashNet`，将高维声学特征压缩为 128 位二进制指纹。
- **高速向量检索层**：采用 Facebook 开源的 `Faiss` 引擎，基于 `IndexBinaryFlat` 算法实现针对 20 万+ 二进制哈希码的毫秒级汉明距离（Hamming Distance）极速匹配。
- **全栈 Web 后端**：`FastAPI` + `Uvicorn` 作为高性能异步服务器，处理路由分发与 CORS；`SQLite` 作为关系型数据存储歌曲元数据、用户档案、收藏夹与识别历史；引入 `bcrypt` 做密码哈希 + `PyJWT` 实现无状态 7 天有效期访问令牌；所有敏感端点（收藏、历史）均通过 `get_current_user(authorization)` 从 JWT payload 提取 `user_id`，**不信任前端传值**，从根本上阻断水平越权。
- **前端交互层**：`Vue 3` 组合式 API 单文件 SPA（严格零构建，CDN 引入），结合 `Axios` 异步请求、`WaveSurfer.js` 动态波形、`Paper Shaders`（基于 WebGPU）的 **Solar Silk 1** 着色器品牌背景；同时在 `visibilitychange` 事件下自动销毁 WebGPU shader 释放 GPU，保证笔记本与移动设备的功耗友好。

## 核心算法难点与破局（科研与工程亮点）

本项目完整经历了深度度量学习（Deep Metric Learning）的典型科研踩坑与优化过程：

1. **时间平移敏感性与滑动窗口**：针对 CNN 在音频时间偏移（截取片段位置不佳）下特征突变的软肋，设计了 **"滑动窗口选秀法"**：在一段音频中提取 RMS（均方根）能量最饱满的片段代表全曲，对抗时间对齐误差。该策略在 `search.py`（在线查询）与 `batch_extract.py`（库构建）中严格对称复用，以确保特征流同构。
2. **哈希空间坍塌与动态中值哈希**：在使用 Triplet Loss 优化距离时曾出现所有输出坍塌为同质化哈希码（全部变成 `111...111`）的致命 Bug。通过引入 **"自适应动态中值哈希"** —— `binary_code = (hash_out > hash_out.median())` —— 替代传统 `> 0` 的死板二值化阈值，强制每条 128 位哈希码严格 0/1 各占 50%，从信息熵角度最大化了哈希编码容量，彻底恢复了哈希指纹的区分度。**该策略是全系统的生命线，严禁被改动。**
3. **领域鸿沟（Domain Gap）与抗噪鲁棒性**：详细论证了干净训练数据集与真实外录场景（设备底噪、频段丢失、房间混响）的分布差异。结合端点检测去静音（VAD）、音量归一化、以及白噪声在线增强与 SpecAugment 等技术，大幅度提升模型在加性噪声条件下的泛化能力。
4. **损失函数的深层次理论探索**：系统核心采用 **大 Margin（5.0）Triplet Margin Loss** + 动态中值阈值的组合，并在理论层面深入探讨了 Quantization Loss（量化惩罚损失）、ITQ（迭代量化）与 STE（Straight-Through Estimator）等前沿 SOTA 方案，为论文的《系统优化探索》及展望章节构筑了深厚的科研壁垒。

## 工程化落地里程碑

除算法突破外，项目同步完成了从"实验代码"到"上线产品"的全套工程化改造：

1. **JWT 无状态鉴权体系**：废弃早期基于 cookie-session 的简易方案，引入 `PyJWT` HS256 签发 7 天有效期 Bearer Token；所有涉及用户数据的端点（`/favorite/toggle`、`/favorites/{uid}`、`/history`）强制走 `get_current_user()` 提取身份，避免水平越权。
2. **速率限制与测试覆盖**：`tests/` 目录搭载 4 个测试文件（`conftest.py` + `test_auth.py` + `test_jwt.py` + `test_preprocessing.py`），覆盖注册登录边界条件、JWT 签发/过期/伪造、音频预处理管线、防刷速率限制等关键契约。
3. **识别历史时间胶囊**：`search_history` 表记录用户每次查询的源音频、命中结果、汉明距离、发生时间，前端通过 `GET /history` 展示可回溯列表、`DELETE /history` 支持一键清空。
4. **持久化迷你播放器**：前端实现一个跟随路由持续存在的底部播放控件，结合 `WaveSurfer.js` 动态波形与 `localStorage` 持久化播放状态，提供沉浸式试听。
5. **VECTORTUNE 品牌化**：定制专属 Logo（`vectortun.svg`）、favicon、Montserrat Variable 字体，并使用 WebGPU 着色器（Solar Silk 1）渲染品牌级动态背景。
6. **Docker 容器化 + Caddy 自动 HTTPS**：`Dockerfile` 基于 `python:3.11-slim` 离线安装 337 MB 预下载 wheels（彻底绕开服务器龟速境外网络），`docker-compose.yml` 编排 `music-search` + `caddy_proxy` 双容器，Caddy 自动为 `evanesce.site` 申请与续期 Let's Encrypt 证书，用户访问全程 HTTPS。
7. **训练加速基础设施**：`precompute_cache.py` 将 MP3/WAV 预解码为 NumPy 波形缓存（`.npy`），消除训练时最昂贵的 `librosa.load` 解码开销，使 GPU 迭代速率提升数倍。

## 当前系统能力边界

- **纯净数字域**：在数字原曲 WAV/MP3 检索上可达 **0 bit 完美暴击命中**，Faiss `IndexBinaryFlat` 对 20.32 万条子指纹的查询稳定在毫秒级。
- **加性噪声域**：对白噪声（SNR 0–20 dB）场景保持高命中率，已通过在线增强 + 单元测试覆盖。
- **物理外录域（未完全攻克）**：对设备底噪 + 房间混响 + 频段扭曲的跨域复合挑战仍有鲁棒性缺口。该方向属深度学习跨域检索的开放性研究问题，已纳入《基于音频数据增强的未来展望》章节。

**学术与商业定位**：本项目最终定位为 **"海量数字音乐版权查重与精准特征推荐系统"**，而非通用"听歌识曲"工具。在数字原曲域已实现生产可用级别的检索精度，并配合 Faiss 达到工业级响应速度，完美覆盖数字库查重与流派相似度推荐两大核心商业指标。
