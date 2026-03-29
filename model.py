import torch
import torch.nn as nn


class AudioHashNet(nn.Module):
    def __init__(self, hash_length=128):
        super(AudioHashNet, self).__init__()

        # 🧠 大脑皮层升级：更宽（通道达到256）、更深（4层卷积）
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # 👑 终极输出层
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # 猛药：随机让一半神经元失忆，防止它死记硬背！
            nn.Linear(256 * 4 * 4, hash_length),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x