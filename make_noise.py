import logging
import os
import sys
import random

import librosa
import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("make_noise")


def generate_noisy_sample(source_path=None, noise_level=0.05):
    if source_path is None:
        raw_dir = "./data/raw"
        music_files = []
        for root, dirs, files in os.walk(raw_dir):
            for f in files:
                if f.lower().endswith(('.mp3', '.wav')):
                    music_files.append(os.path.join(root, f))
        if not music_files:
            logger.error("未找到任何音频文件，请先放置数据到 %s", raw_dir)
            return
        source_path = random.choice(music_files)
        logger.info("随机选取源文件: %s", source_path)

    if not os.path.exists(source_path):
        logger.error("源文件不存在: %s", source_path)
        return

    y, sr = librosa.load(source_path, sr=22050)
    noise = np.random.randn(len(y))
    y_noisy = y + noise_level * noise

    base = os.path.splitext(os.path.basename(source_path))[0]
    output_path = f"{base}_noisy_test.wav"
    sf.write(output_path, y_noisy, sr)
    logger.info("成功生成带噪音测试文件: %s", os.path.abspath(output_path))


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else None
    noise_lvl = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    generate_noisy_sample(source, noise_lvl)
