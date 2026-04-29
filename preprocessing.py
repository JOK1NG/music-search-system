import librosa
import numpy as np
import torch

from config import N_MELS, SAMPLE_RATE, WINDOW_DURATION


def load_audio(file_path, duration=None):
    return librosa.load(file_path, sr=SAMPLE_RATE, duration=duration)


def pad_audio(y, target_duration, sr=SAMPLE_RATE):
    target_len = int(sr * target_duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y[:target_len]


def trim_silence(y, top_db=30, sr=SAMPLE_RATE):
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if len(y_trimmed) < int(sr * 0.5):
        return y
    return y_trimmed


def build_mel_spectrogram(y, sr=SAMPLE_RATE):
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    return librosa.power_to_db(spec, ref=np.max)


def spec_to_tensor(spec_db, device=None):
    tensor = torch.FloatTensor(spec_db).unsqueeze(0).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def binarize_hash(hash_out):
    threshold = hash_out.median()
    binary_code = (hash_out > threshold).int().cpu().numpy()
    return np.packbits(binary_code, axis=1).astype("uint8")


def slice_best_windows(y, window_duration=WINDOW_DURATION, n_windows=3, sr=SAMPLE_RATE):
    total_duration = len(y) / sr
    if total_duration < window_duration:
        y = pad_audio(y, window_duration, sr=sr)
        total_duration = window_duration

    window_candidates = []
    for start_sec in range(0, max(1, int(total_duration - window_duration + 1))):
        start_sample = int(start_sec * sr)
        end_sample = int((start_sec + window_duration) * sr)
        y_window = y[start_sample:end_sample]
        y_window = pad_audio(y_window, window_duration, sr=sr)
        energy = np.mean(librosa.feature.rms(y=y_window))
        window_candidates.append((energy, y_window, start_sec))

    window_candidates.sort(key=lambda item: item[0], reverse=True)
    return window_candidates[:n_windows]


def extract_hash_from_window(y_window, model, device, sr=SAMPLE_RATE):
    spec_db = build_mel_spectrogram(y_window, sr=sr)
    tensor = spec_to_tensor(spec_db, device=device)
    with torch.no_grad():
        hash_out = model(tensor)
    return binarize_hash(hash_out)
