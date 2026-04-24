import numpy as np
import librosa
import torch
from config import SAMPLE_RATE, N_MELS, WINDOW_DURATION


def load_audio(file_path, duration=None):
    """加载音频文件，返回波形与采样率"""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=duration)
    return y, sr


def pad_audio(y, target_duration):
    """将短音频补零至目标时长"""
    if target_duration is None:
        return y
    target_len = int(SAMPLE_RATE * target_duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y[:target_len]


def build_mel_spectrogram(y, sr=None):
    """提取梅尔频谱并转换为 dB —— 统一接口"""
    if sr is None:
        sr = SAMPLE_RATE
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db


def binarize_hash(hash_out):
    """动态中值哈希二值化 —— 与建库/检索绝对一致"""
    threshold = hash_out.median()
    binary_code = (hash_out > threshold).int().cpu().numpy()
    return np.packbits(binary_code, axis=1).astype('uint8')


def slice_best_windows(y, window_duration=WINDOW_DURATION, n_windows=3):
    """从完整波形中按 RMS 能量选取最佳的 n 个窗口"""
    sr = SAMPLE_RATE
    total_duration = len(y) / sr

    if total_duration < window_duration:
        y = pad_audio(y, window_duration)
        total_duration = window_duration

    window_candidates = []
    for start_sec in range(0, max(1, int(total_duration - window_duration + 1))):
        start_sample = int(start_sec * sr)
        end_sample = int((start_sec + window_duration) * sr)
        y_window = y[start_sample:end_sample]

        if len(y_window) < int(sr * window_duration):
            y_window = np.pad(y_window, (0, int(sr * window_duration) - len(y_window)))

        energy = np.mean(librosa.feature.rms(y=y_window))
        window_candidates.append((energy, y_window, start_sec))

    window_candidates.sort(key=lambda x: x[0], reverse=True)
    return window_candidates[:n_windows]


def spec_to_tensor(spec_db, device=None):
    """将频谱转为适合模型的 tensor"""
    tensor = torch.FloatTensor(spec_db).unsqueeze(0).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def extract_hash_from_window(y_window, model, device):
    """对单个窗口提取二值哈希"""
    spec_db = build_mel_spectrogram(y_window)
    tensor = spec_to_tensor(spec_db, device)
    with torch.no_grad():
        hash_out = model(tensor)
    return binarize_hash(hash_out)


def extract_hash_from_full_spectrum(spec_db, model, device, start_sec, window_duration=WINDOW_DURATION, sr=SAMPLE_RATE):
    """从完整频谱中按时间轴切片并提取二值哈希（复用已计算的频谱）"""
    hop_length = 512
    frames_per_window = int((window_duration * sr) / hop_length)
    start_frame = int((start_sec * sr) / hop_length)
    end_frame = start_frame + frames_per_window

    spec_slice = spec_db[:, start_frame:end_frame]
    tensor = spec_to_tensor(spec_slice, device)
    with torch.no_grad():
        hash_out = model(tensor)
    return binarize_hash(hash_out)
