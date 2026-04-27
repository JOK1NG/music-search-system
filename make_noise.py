import librosa
import numpy as np
import soundfile as sf
import os

# 1. 读取原版完美音频
original_path = "./data/raw/000/000002.mp3"  # 确保这个路径是对的
y, sr = librosa.load(original_path, sr=22050)

# 2. 制造软件级别的白噪声（模拟电流麦、背景底噪）
noise = np.random.randn(len(y))
# ⚠️ 必须与 dataset.py 的训练增强分布对齐（训练用 0.01）：
# - 0.01 ≈ 24 dB SNR，完美命中；0.02 ≈ 18 dB SNR，hamming≈6，rank1 稳得住；
# - 0.03+ 超出训练分布，模型哈希开始失守，真命中会掉出 top5。
y_noisy = y + 0.02 * noise

# 3. 保存为带噪音的测试文件
output_path = "000002_noisy_test.wav"
sf.write(output_path, y_noisy, sr)

print(f"✅ 成功生成带噪音的测试文件: {os.path.abspath(output_path)}")
print("快去网页上把这个文件传上去测一下！")

#todo 可否实现随机选取文件进行加噪音展示 ？