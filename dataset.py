import os
import random
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader


class MusicTripletDataset(Dataset):
    def __init__(self, raw_path, duration=5.0, sr=22050):
        self.raw_path = raw_path
        self.duration = duration
        self.sr = sr
        self.music_files = self._get_files()

    def _get_files(self):
        files = []
        for root, dirs, filenames in os.walk(self.raw_path):
            for f in filenames:
                if f.lower().endswith('.mp3'):
                    files.append(os.path.join(root, f))
        return files

    def __len__(self):
        return len(self.music_files)

    def _extract_mel(self, y):
        # 提取梅尔频谱 [cite: 728-732]
        spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        return torch.FloatTensor(spec_db).unsqueeze(0)  # [1, Height, Width]

    def _add_noise(self, y, noise_factor=0.005):
        # 数据增强：人为添加白噪声，模拟环境底噪
        noise = np.random.randn(len(y))
        augmented_data = y + noise_factor * noise
        return augmented_data

    def __getitem__(self, idx):
        # 1. 获取基准歌曲 (Anchor)
        anchor_path = self.music_files[idx]

        # 2. 随机抽取另一首完全不同的歌 (Negative)
        neg_idx = random.randint(0, len(self.music_files) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.music_files) - 1)
        negative_path = self.music_files[neg_idx]

        try:
            # 加载 Anchor 原始音频
            y_anchor, _ = librosa.load(anchor_path, sr=self.sr, duration=self.duration)

            # 制造 Positive 音频：给 Anchor 加点噪音
            y_positive = self._add_noise(y_anchor)

            # 加载 Negative 原始音频
            y_negative, _ = librosa.load(negative_path, sr=self.sr, duration=self.duration)

            # 如果某些歌太短不够长，直接补零对齐尺寸
            target_length = int(self.sr * self.duration)
            y_anchor = np.pad(y_anchor, (0, max(0, target_length - len(y_anchor))))[:target_length]
            y_positive = np.pad(y_positive, (0, max(0, target_length - len(y_positive))))[:target_length]
            y_negative = np.pad(y_negative, (0, max(0, target_length - len(y_negative))))[:target_length]

            # 转换为频谱图矩阵
            spec_anchor = self._extract_mel(y_anchor)
            spec_positive = self._extract_mel(y_positive)
            spec_negative = self._extract_mel(y_negative)

            return spec_anchor, spec_positive, spec_negative

        except Exception as e:
            # 如果某首歌坏了，就递归随便再取一首保底
            return self.__getitem__(random.randint(0, len(self.music_files) - 1))


# 测试一下我们写的数据集
if __name__ == "__main__":
    test_dataset = MusicTripletDataset("./data/raw")
    print(f"📦 数据集初始化成功！包含 {len(test_dataset)} 首歌曲。")

    # 抽取第一组三元组看看
    a, p, n = test_dataset[0]
    print(f"📐 Anchor 维度: {a.shape}")
    print(f"📐 Positive 维度: {p.shape}")
    print(f"📐 Negative 维度: {n.shape}")