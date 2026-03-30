import os
import random
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset

# 🌟 波形缓存目录（由 precompute_cache.py 生成）
WAVEFORM_CACHE_PATH = "./data/waveform_cache"

class AudioTripletDataset(Dataset):
    def __init__(self, raw_path, duration=5.0, sr=22050):
        self.raw_path = raw_path
        self.duration = duration
        self.sr = sr
        self.music_files = self._get_files()
        self.use_cache = os.path.isdir(WAVEFORM_CACHE_PATH)
        if self.use_cache:
            print(f"⚡ 检测到波形缓存目录，将优先从缓存加载（跳过 MP3 解码）")
        else:
            print(f"⚠️ 未找到波形缓存，将实时解码音频（可运行 precompute_cache.py 加速）")

    def _get_files(self):
        files = []
        for root, dirs, filenames in os.walk(self.raw_path):
            for f in filenames:
                if f.lower().endswith(('.mp3', '.wav')):
                    files.append(os.path.join(root, f))
        return files

    def __len__(self):
        return len(self.music_files)

    def _load_waveform(self, file_path):
        """优先从缓存加载波形，缓存未命中则回退到 librosa 实时解码"""
        target_length = int(self.sr * self.duration)

        if self.use_cache:
            rel_path = os.path.relpath(file_path, self.raw_path)
            cache_name = rel_path.replace(os.sep, "__") + ".npy"
            cache_file = os.path.join(WAVEFORM_CACHE_PATH, cache_name)
            if os.path.exists(cache_file):
                return np.load(cache_file)

        # 缓存未命中，回退到 librosa（兼容未预计算的情况）
        y, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        return y[:target_length]

    def _extract_mel(self, y):
        spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        return torch.FloatTensor(spec_db).unsqueeze(0)

    # 🌟 修改点：噪音降回健康的 0.01
    def _add_noise(self, y, noise_factor=0.01):
        noise = np.random.randn(len(y))
        augmented_data = y + noise_factor * noise
        return augmented_data

    def __getitem__(self, idx):
        anchor_path = self.music_files[idx]
        neg_idx = random.randint(0, len(self.music_files) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.music_files) - 1)
        negative_path = self.music_files[neg_idx]

        try:
            y_anchor = self._load_waveform(anchor_path)
            y_positive = self._add_noise(y_anchor)
            y_negative = self._load_waveform(negative_path)

            spec_anchor = self._extract_mel(y_anchor)
            spec_positive = self._extract_mel(y_positive)
            spec_negative = self._extract_mel(y_negative)

            return spec_anchor, spec_positive, spec_negative
        except Exception as e:
            return self.__getitem__(random.randint(0, len(self.music_files) - 1))