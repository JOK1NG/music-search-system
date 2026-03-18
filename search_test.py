import faiss
import numpy as np

# 1. 加载打包后的数据
fingerprints = np.load("./data/processed/all_music_hashes.npy")
mapping = np.load("./data/processed/file_mapping.npy")

# 2. 初始化索引：128 是指位宽 (Bits)，不是字节数
index = faiss.IndexBinaryFlat(128)

# 3. 添加数据 (形状必须是 [N, 16], 类型必须是 uint8)
index.add(fingerprints)
print(f"✅ 索引已就绪，当前库内歌曲数: {index.ntotal}")

# 4. 模拟检索第一首歌
query = fingerprints[0:1]
D, I = index.search(query, k=3)

print("\n🔍 检索结果：")
for i in range(len(I[0])):
    idx = I[0][i]
    if idx != -1: # -1 表示没搜到
        print(f"排名 {i+1}: {mapping[idx]} (差异: {D[0][i]} 位)")