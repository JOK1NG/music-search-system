"""
search.py —— 在线检索大脑
----------------------------------
职责：接收一段待检测音频（由前端经 main.py 上传），将其在线提取为 128bit 哈希指纹，
     再调用 Faiss 二进制索引在十万级子指纹库中做毫秒级汉明距离匹配，最后返回 Top 5 候选。

与离线建库脚本 batch_extract.py 严格对称：
  - 采样率 22050、单段时长 5.0s、n_mels=128 必须完全一致；
  - 二值化阈值必须使用动态中值 (`> hash_out.median()`)，禁止改回 `> 0`，
    否则会触发哈希空间坍塌 + Domain Shift（详见 AGENTS.md 安全栏 #2）。
"""

import os                                # 仅备扩展用（路径处理），当前未直接调用
import torch                              # 深度学习推理引擎，承载 AudioHashNet 前向
import librosa                            # 音频 DSP 工具：解码、Mel 频谱、RMS 能量
import numpy as np                        # 张量↔ndarray 互转 & packbits 比特打包
import sqlite3                            # 通过 faiss_id 反查歌曲元数据（标题/作者/路径）
import faiss                              # 向量检索引擎，使用 IndexBinaryFlat 做汉明距离
from model import AudioHashNet            # 自研 CNN 深度哈希网络（128bit 输出）

# --- ⚙️ 全局配置参数 ---
# SQLite 元数据库：存 songs 表（faiss_id ↔ title/artist/file_path 映射）
DB_PATH = "music_hash.db"
# Faiss 二进制索引文件：保存全库子指纹（128bit 一行），由 batch_extract.py 离线生成
INDEX_PATH = "music_hash.index"
# 模型权重路径：⚠️ 第 3 轮是经过 Triplet Loss 完美收敛的最佳模型，不要随意切换
MODEL_PATH = "./checkpoints/hash_model_epoch_3.pth"  # ⚠️ 确保使用的是完美收敛的第3轮模型


def load_system():
    """
    系统冷启动：把 Faiss 索引与深度哈希模型一次性加载进内存。
    本函数被 main.py 在 FastAPI 应用启动阶段调用一次（lifespan / startup），
    避免每次 /search 请求都重复 IO 与模型反序列化，保证毫秒级响应。

    返回:
        (index, model)
          - index: faiss.IndexBinaryFlat，已 mmap/读入内存的子指纹库
          - model: AudioHashNet，已切到 eval() 模式的推理网络
        任一加载失败时对应位置返回 None，由调用方决定是否降级。
    """
    print("⏳ 正在启动检索引擎与加载 8 缸 AI 大脑...")
    # ---- ① 读取 Faiss 二进制索引（IndexBinaryFlat：暴力汉明距离，零误差）----
    try:
        index = faiss.read_index_binary(INDEX_PATH)
        print(f"✅ Faiss 索引库加载成功！当前库中包含 {index.ntotal} 条子指纹数据。")
    except Exception as e:
        # 索引文件缺失通常意味着尚未跑过 batch_extract.py，直接早退
        print(f"❌ Faiss 索引加载失败: {e}")
        return None, None

    # ---- ② 选择推理设备：有 GPU 用 GPU，否则退回 CPU（Docker 部署默认 CPU）----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioHashNet().to(device)

    # ---- ③ 反序列化权重并切到 eval 模式（关闭 Dropout/BN 训练态）----
    try:
        # map_location=device 让在 GPU 训练的权重也能在 CPU 上无缝加载
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("✅ 深度哈希模型权重加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return index, None

    return index, model


def search_music(test_audio_path, index, model, top_k=150):
    """
    在线检索主入口：对单段待检测音频做切片 → 哈希 → Faiss 检索 → 元数据回填。

    参数:
        test_audio_path: 前端上传文件落盘后的临时路径（任意 librosa 可解码格式）
        index:           load_system() 返回的 Faiss 二进制索引
        model:           load_system() 返回的 AudioHashNet 实例
        top_k:           每个查询切片向 Faiss 索要的近邻数（默认 150，覆盖率 vs 速度的折中）

    返回:
        list[dict]，长度 ≤ 5；每项含 rank/title/artist/distance/file_path/audio_url
    """
    # ==========================================
    # 🌟 第一步：音频预处理 (素颜状态，不加特效，与建库绝对一致)
    # ==========================================
    # sr=22050：与 batch_extract.py 完全一致，重采样保证频域特征对齐
    # duration=15.0：只取前 15 秒，控制最大计算量并避免长音频拖慢响应
    y, sr = librosa.load(test_audio_path, sr=22050, duration=15.0)

    # 极短音频保护：不足 5s 的样本 0-pad 到 5s，避免下方滑动窗口取不到一帧
    total_duration = len(y) / sr
    if total_duration < 5.0:
        y = np.pad(y, (0, int(sr * 5.0) - len(y)))
        total_duration = 5.0

    # 通过任意一个参数推断当前模型所在设备（CPU/GPU），后续张量需 to(device)
    device = next(model.parameters()).device

    # ==========================================
    # 🌟 第二步：能量选秀！只取声音最饱满的前3个切片
    # ==========================================
    # 设计动机：音频开头/结尾常含静音、淡入淡出，直接拿首段切片做查询很容易撞库失败。
    # 这里以 1s 为步长滑动取若干 5s 窗，按 RMS（均方根值（Root Mean Square））能量降序排序，挑能量最高的 Top 3 当查询，
    # 相当于做"安静段过滤"，显著提升对真实人声/前奏不一致样本的鲁棒性。
    window_candidates = []
    # max(1, ...) 兜底：当 total_duration ≤ 5 时仍至少跑一次循环
    for start_sec in range(0, max(1, int(total_duration - 4))):
        start_sample = int(start_sec * sr)
        end_sample = int((start_sec + 5.0) * sr)
        y_window = y[start_sample:end_sample]

        # 末窗可能不足 5s（音频尾部），同样 0-pad 到固定长度，确保 Mel 频谱形状统一
        if len(y_window) < int(sr * 5.0):
            y_window = np.pad(y_window, (0, int(sr * 5.0) - len(y_window)))

        # RMS（Root Mean Square）= 时域能量，对每个窗口取均值作为"响度评分"
        energy = np.mean(librosa.feature.rms(y=y_window))
        window_candidates.append((energy, y_window))

    # 按能量降序：能量大 = 信息量大 = 更利于哈希区分
    window_candidates.sort(key=lambda x: x[0], reverse=True)
    best_windows = window_candidates[:3]  # 仅保留 Top 3，控制下游推理成本

    # 把 3 个最优窗口分别过一遍 CNN，得到 3 个查询哈希码（多查询提升召回）
    queries = []
    for energy, y_window in best_windows:
        # ---- 时域 → 频域：转 Mel 频谱（n_mels=128 与训练严格一致）----
        spec = librosa.feature.melspectrogram(y=y_window, sr=sr, n_mels=128)
        # 功率谱 → 分贝刻度，逼近人耳听觉感知，并压缩动态范围
        spec_db = librosa.power_to_db(spec, ref=np.max)
        # 升 batch 维 (1,) 与通道维 (1,)，得到 [1, 1, 128, T] 形状的 4D 张量
        tensor = torch.FloatTensor(spec_db).unsqueeze(0).unsqueeze(0).to(device)

        # 推理路径关闭梯度计算，省显存提速度
        with torch.no_grad():
            hash_out = model(tensor)  # 输出 [1, 128] 的连续浮点编码

            # 🌟 核心破局魔法：必须和建库保持一致的中值过滤！
            # 解释：早期使用 `> 0` 死阈值时，模型坍塌后输出会全偏一侧，
            # 导致所有样本哈希码几乎相同，全库命中率瞬间归零。
            # 改用动态中值 → 强制每个样本恰好一半比特为 1、一半为 0，
            # 等价于一个"自适应去均值"操作，彻底破解 Domain Shift 与坍塌。
            threshold = hash_out.median()
            binary_code = (hash_out > threshold).int().cpu().numpy()

            # 把 128 个 0/1 比特位打包成 16 字节 uint8，符合 Faiss BinaryFlat 输入规范
            query_packed = np.packbits(binary_code, axis=1).astype('uint8')
            queries.append(query_packed)

    # 极端兜底：若上游全部失败（极罕见），返回空结果让前端显示"未找到"
    if not queries:
        return []

    # ==========================================
    # 🌟 第三步：多重检索与投票合并
    # ==========================================
    # 思路：3 个查询哈希分别独立检索 Top-K，再以 (歌曲文件路径) 为 key 做"歌曲级聚合"。
    # 同一首歌可能被多个窗口同时命中，最终保留汉明距离最小的那一次（最强证据）。
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    song_best_match = {}  # key=file_path，value=最优命中记录字典

    for query_packed in queries:
        # Faiss 二进制检索：返回汉明距离矩阵 D 与索引矩阵 I，形状均为 [1, top_k]
        D, I = index.search(query_packed, k=top_k)

        for i in range(len(I[0])):
            faiss_id = int(I[0][i])
            distance = int(D[0][i])  # 汉明距离：0 表示完美命中，越小越相似

            # Faiss 用 -1 表示"无足够近邻"，必须显式过滤
            if faiss_id != -1:
                # 用 faiss_id 反查 SQLite 元数据（建库时一并写入）
                cursor.execute("SELECT title, artist, file_path FROM songs WHERE faiss_id=?", (faiss_id,))
                row = cursor.fetchone()
                if row:
                    title, artist, file_path = row

                    # --------------------------------------------------------
                    # 🌟 核心修改区：生成前端能播放的网络 URL
                    # 1. 抹平系统差异：把 Windows 本地的反斜杠 '\' 全换成网址用的正斜杠 '/'
                    #    （建库可能在 Windows 跑，部署在 Linux，路径分隔符会混入 \）
                    normalized_path = file_path.replace("\\", "/")

                    # 2. 截断与拼接：截取 'raw/' 后面的具体文件名，拼上咱们在 main.py 开通的 '/audio/' 专线
                    #    main.py 里通过 app.mount("/audio", StaticFiles(directory="data/raw")) 暴露音频
                    if "raw/" in normalized_path:
                        audio_url = "/audio/" + normalized_path.split("raw/")[-1]
                    else:
                        audio_url = ""  # 保底防崩机制（路径异常时给空串，前端会隐藏播放器）
                    # --------------------------------------------------------

                    # 记录并去重，只保留同一首歌里汉明距离最小（最相似）的那一次撞击
                    if file_path not in song_best_match or distance < song_best_match[file_path]["distance"]:
                        song_best_match[file_path] = {
                            "title": title,
                            "artist": artist,
                            "distance": distance,
                            "file_path": file_path,
                            "audio_url": audio_url  # 👈 关键点：把播放链接装进字典返回给前端！
                        }
    conn.close()

    # ==========================================
    # 🌟 第四步：排序并输出最终 Top 5
    # ==========================================
    # 按汉明距离升序：0 bit 完美命中排第一；前端拿到后再做 "确信度/相似度" 展示
    sorted_results = sorted(song_best_match.values(), key=lambda x: x["distance"])

    # 截断 Top 5 并附加 rank 字段，方便前端直接渲染排行榜，无需再算下标
    final_results = []
    for i, res in enumerate(sorted_results[:5]):
        res["rank"] = i + 1
        final_results.append(res)

    return final_results