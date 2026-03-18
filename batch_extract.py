import os
import torch
import numpy as np
import librosa
from model import AudioHashNet

# 1. 路径配置
RAW_PATH = "./data/raw"
SAVE_PATH = "./data/processed"
# 指向你刚刚炼好的“最强大脑”
MODEL_WEIGHTS = "./checkpoints/hash_model_epoch_5.pth"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def get_spec_tensor(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=5.0)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    tensor = torch.FloatTensor(spec_db).unsqueeze(0).unsqueeze(0)
    return tensor


def main():
    print(f"🔍 正在扫描路径: {os.path.abspath(RAW_PATH)}")

    # 🌟 核心升级：加载训练好的模型权重
    model = AudioHashNet()
    if os.path.exists(MODEL_WEIGHTS):
        # 你的电脑是 CPU 训练的，所以加上 map_location='cpu'
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location='cpu', weights_only=True))
        print("🧠 成功加载经过训练的模型权重！系统智商已上线。")
    else:
        print("❌ 错误：找不到权重文件，请确认 train.py 已经跑完了第 5 个 Epoch！")
        return

    model.eval()

    music_files = []
    for root, dirs, files in os.walk(RAW_PATH):
        for file in files:
            if file.lower().endswith('.mp3'):
                music_files.append(os.path.join(root, file))

    if len(music_files) == 0:
        print("❌ 错误：没找到 mp3 文件！")
        return

    print(f"🚀 发现 {len(music_files)} 首歌曲。开始提取高质量指纹...")

    all_hashes = []
    file_mapping = []

    for i, path in enumerate(music_files):
        try:
            spec = get_spec_tensor(path)
            with torch.no_grad():
                hash_out = model(spec)
                binary_code = (hash_out > 0).int().numpy()
                all_hashes.append(binary_code)
                file_mapping.append(path)

            if i % 10 == 0:
                print(f"⌛ 已处理: {i}/{len(music_files)}...")
        except Exception as e:
            pass  # 遇到坏文件直接静默跳过

    # 2. 保存新的指纹库
    if all_hashes:
        final_data = np.vstack(all_hashes).reshape(len(all_hashes), -1)
        packed_hashes = np.packbits(final_data, axis=1).astype('uint8')

        np.save(f"{SAVE_PATH}/all_music_hashes.npy", packed_hashes)
        np.save(f"{SAVE_PATH}/file_mapping.npy", np.array(file_mapping))

        print("\n" + "=" * 30)
        print(f"✅ 高质量指纹提取圆满完成！旧的记忆已被覆盖。")
        print("=" * 30)


if __name__ == "__main__":
    main()