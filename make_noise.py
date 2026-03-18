import librosa
import numpy as np
import soundfile as sf
import os

# 1. 读取原版完美音频
original_path = "./data/raw/000/000002.mp3"  # 确保这个路径是对的
y, sr = librosa.load(original_path, sr=22050)

# 2. 制造软件级别的白噪声（模拟电流麦、背景底噪）
noise = np.random.randn(len(y))
# 控制噪声大小，0.05 已经是很明显的沙沙声了
y_noisy = y + 0.05 * noise

# 3. 保存为带噪音的测试文件
output_path = "000002_noisy_test.wav"
sf.write(output_path, y_noisy, sr)

print(f"✅ 成功生成带噪音的测试文件: {os.path.abspath(output_path)}")
print("快去网页上把这个文件传上去测一下！")

#todo 可否实现随机选取文件进行加噪音展示 ？