import os
import random
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset

class AudioTripletDataset(Dataset):
    def __init__(self, raw_path, duration=5.0, sr=22050):
        self.raw_path = raw_path
        self.duration = duration
        self.sr = sr
        self.music_files = self._get_files()

    def _get_files(self):
        files = []
        for root, dirs, filenames in os.walk(self.raw_path):
            for f in filenames:
                if f.lower().endswith(('.mp3', '.wav')):
                    files.append(os.path.join(root, f))
        return files

    def __len__(self):
        return len(self.music_files)

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
            y_anchor, _ = librosa.load(anchor_path, sr=self.sr, duration=self.duration)
            y_positive = self._add_noise(y_anchor)
            y_negative, _ = librosa.load(negative_path, sr=self.sr, duration=self.duration)

            target_length = int(self.sr * self.duration)
            if len(y_anchor) < target_length:
                y_anchor = np.pad(y_anchor, (0, target_length - len(y_anchor)))
            y_anchor = y_anchor[:target_length]
            if len(y_positive) < target_length:
                y_positive = np.pad(y_positive, (0, target_length - len(y_positive)))
            y_positive = y_positive[:target_length]
            if len(y_negative) < target_length:
                y_negative = np.pad(y_negative, (0, target_length - len(y_negative)))
            y_negative = y_negative[:target_length]

            spec_anchor = self._extract_mel(y_anchor)
            spec_positive = self._extract_mel(y_positive)
            spec_negative = self._extract_mel(y_negative)

            return spec_anchor, spec_positive, spec_negative
        except Exception as e:
            return self.__getitem__(random.randint(0, len(self.music_files) - 1))