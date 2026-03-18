import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def get_spec(file_path):
    # 1. 加载音频 (统一采样率 22050Hz)
    y, sr = librosa.load(file_path, sr=22050, duration=5.0)  # 只取前5秒测试

    # 2. 提取梅尔频谱
    # n_mels=128 是常用标准
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    # 3. 转为分贝刻度（让特征更明显）
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 可视化看看（这就是喂给 CNN 的“图片”）
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spectrogram')
    plt.show()
    return S_dB


if __name__ == "__main__":
    # 1. 这里填你电脑上真实的音乐文件路径
    # 注意：把 YourName 换成你电脑的用户名，或者直接填一个简单的路径
    target_path = "Derra,Drake,Rihanna - Work (Derra Flip).mp3"

    # 2. 调用上面定义的函数
    print("正在处理音频，请稍候...")
    result = get_spec(target_path)
    print("处理完成，矩阵形状为:", result.shape)
