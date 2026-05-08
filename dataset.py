import os
import random
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset

# 🌟 波形缓存：旧 5s 缓存仅给推理脚本用，训练用新的 30s 缓存
WAVEFORM_CACHE_30S_PATH = "./data/waveform_cache_30s"
FULL_DURATION = 30.0  # 与 batch_extract.py 的滑动窗口区间保持一致（前 30s）
WINDOW_DURATION = 5.0  # 单个训练样本时长


class AudioTripletDataset(Dataset):
    """
    Triplet 数据集，专为破解"哈希空间坍塌"设计：
      - anchor / positive 来自同一首歌的两个不同随机 5s 起点（时间偏移不变性）
      - negative 来自另一首歌的随机 5s 起点
      - 在频域叠加 SpecAugment + 音量缩放 + 轻量白噪 三件套
    """

    def __init__(self, raw_path, sr=22050, full_duration=FULL_DURATION,
                 window_duration=WINDOW_DURATION, augment=True):
        self.raw_path = raw_path
        self.sr = sr
        self.full_duration = full_duration
        self.window_duration = window_duration
        self.window_samples = int(sr * window_duration)
        self.full_samples = int(sr * full_duration)
        self.augment = augment

        self.music_files = self._get_files()
        self.use_cache_30s = os.path.isdir(WAVEFORM_CACHE_30S_PATH)
        if self.use_cache_30s:
            print(f"⚡ 检测到 30s 波形缓存，训练时跳过 MP3 解码")
        else:
            print(f"⚠️ 未找到 30s 波形缓存，将实时解码（建议跑 precompute_cache.py --duration=30 加速）")

    def _get_files(self):
        files = []
        for root, dirs, filenames in os.walk(self.raw_path):
            for f in filenames:
                if f.lower().endswith(('.mp3', '.wav')):
                    files.append(os.path.join(root, f))
        return files

    def __len__(self):
        return len(self.music_files)

    # ----------------------------------------------------------
    # 波形加载
    # ----------------------------------------------------------
    def _load_full_waveform(self, file_path):
        """读取前 30s 完整波形；优先 30s 缓存，缺失则实时解码并补零到固定长度"""
        if self.use_cache_30s:
            rel_path = os.path.relpath(file_path, self.raw_path)
            cache_name = rel_path.replace(os.sep, "__") + ".npy"
            cache_file = os.path.join(WAVEFORM_CACHE_30S_PATH, cache_name)
            if os.path.exists(cache_file):
                y = np.load(cache_file)
                if len(y) < self.full_samples:
                    y = np.pad(y, (0, self.full_samples - len(y)))
                return y[:self.full_samples]

        y, _ = librosa.load(file_path, sr=self.sr, duration=self.full_duration)
        if len(y) < self.full_samples:
            y = np.pad(y, (0, self.full_samples - len(y)))
        return y[:self.full_samples]

    def _random_window(self, y_full):
        """从前 30s 随机抽取一个 5s 起点"""
        max_start = max(0, len(y_full) - self.window_samples)
        start = random.randint(0, max_start) if max_start > 0 else 0
        return y_full[start:start + self.window_samples]

    # ----------------------------------------------------------
    # 数据增强（频域 + 波形域）
    # ----------------------------------------------------------
    def _add_noise(self, y, noise_factor=0.01):
        return y + noise_factor * np.random.randn(len(y)).astype(y.dtype)

    def _random_gain(self, y, db_range=6.0):
        """随机音量缩放 ±db_range（默认 ±6dB）"""
        gain_db = random.uniform(-db_range, db_range)
        return y * (10 ** (gain_db / 20.0))

    def _spec_augment(self, spec_db, freq_mask_param=20, time_mask_param=30,
                      n_freq_masks=2, n_time_masks=2):
        """SpecAugment 轻量版：在 Mel 频谱上随机遮蔽若干频带和时间段"""
        spec = spec_db.clone()
        n_mels, n_frames = spec.shape[-2], spec.shape[-1]
        # 频率 mask
        for _ in range(n_freq_masks):
            f = random.randint(0, freq_mask_param)
            f0 = random.randint(0, max(0, n_mels - f))
            spec[..., f0:f0 + f, :] = spec.min()
        # 时间 mask
        for _ in range(n_time_masks):
            t = random.randint(0, time_mask_param)
            t0 = random.randint(0, max(0, n_frames - t))
            spec[..., :, t0:t0 + t] = spec.min()
        return spec

    # ----------------------------------------------------------
    # 频谱提取
    # ----------------------------------------------------------
    def _extract_mel(self, y):
        spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        return torch.FloatTensor(spec_db).unsqueeze(0)

    def _build_sample(self, y_window, apply_augment):
        """5s 波形 → 增强 → Mel 频谱 (1, 128, T)"""
        if apply_augment and self.augment:
            y_window = self._random_gain(y_window)
            y_window = self._add_noise(y_window, noise_factor=0.01)
        spec = self._extract_mel(y_window)
        if apply_augment and self.augment:
            spec = self._spec_augment(spec)
        return spec

    # ----------------------------------------------------------
    # Triplet 取样
    # ----------------------------------------------------------
    def __getitem__(self, idx):
        anchor_path = self.music_files[idx]
        neg_idx = random.randint(0, len(self.music_files) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.music_files) - 1)
        negative_path = self.music_files[neg_idx]

        try:
            y_anchor_full = self._load_full_waveform(anchor_path)
            y_negative_full = self._load_full_waveform(negative_path)

            # 同一首歌的两个不同 5s 起点 → 学习"时间偏移不变性"
            y_anchor = self._random_window(y_anchor_full)
            y_positive = self._random_window(y_anchor_full)
            y_negative = self._random_window(y_negative_full)

            # v2: anchor 保持原始分布（接近推理时用户上传的 query），
            #     只对 positive 施加 gain + noise + SpecAugment，
            #     让模型真正学到"增强后版本与原始版本是同一首歌"的不变性.
            #     v1 两边都增强，导致 anchor/positive 都偏离推理分布，模型学偏.
            spec_anchor = self._build_sample(y_anchor, apply_augment=False)
            spec_positive = self._build_sample(y_positive, apply_augment=True)
            spec_negative = self._build_sample(y_negative, apply_augment=True)

            return spec_anchor, spec_positive, spec_negative
        except Exception:
            return self.__getitem__(random.randint(0, len(self.music_files) - 1))