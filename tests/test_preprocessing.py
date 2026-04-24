import numpy as np
import pytest
from preprocessing import (
    pad_audio,
    build_mel_spectrogram,
    binarize_hash,
    slice_best_windows,
)


def test_pad_audio():
    import numpy as np
    sr = 22050
    short = np.zeros(int(sr * 2), dtype=np.float32)
    padded = pad_audio(short, 5.0)
    assert len(padded) == int(sr * 5.0)


def test_build_mel_spectrogram():
    import numpy as np
    sr = 22050
    y = np.random.randn(int(sr * 3)).astype(np.float32)
    spec = build_mel_spectrogram(y, sr)
    assert spec.shape[0] == 128
    assert spec.shape[1] > 0


def test_binarize_hash():
    import torch
    hash_out = torch.rand(1, 128)
    packed = binarize_hash(hash_out)
    assert packed.shape == (1, 16)
    assert packed.dtype == np.uint8


def test_slice_best_windows():
    import numpy as np
    sr = 22050
    y = np.random.randn(int(sr * 10)).astype(np.float32)
    windows = slice_best_windows(y, window_duration=5.0, n_windows=3)
    assert len(windows) == 3
    for energy, y_window, start_sec in windows:
        assert len(y_window) == int(sr * 5.0)
        assert start_sec >= 0
