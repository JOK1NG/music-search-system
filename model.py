import torch
import torch.nn as nn


class AudioHashNet(nn.Module):
    def __init__(self, hash_length=128):
        super(AudioHashNet, self).__init__()

        # 👑 第一层：提取浅层纹理
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 猛药1：批归一化防崩溃
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # 👑 第二层：提取中层声学特征
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # 👑 第三层：提取深层语义风格
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        # 自适应池化：不管输入的音频切片差了几毫秒，强制转换成 4x4 大小，极度增强鲁棒性！
        self.adap_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 👑 终极输出层：映射到 128 位哈希空间
        self.fc = nn.Linear(128 * 4 * 4, hash_length)
        self.tanh = nn.Tanh()  # 猛药2：把输出死死限制在 -1 到 1 之间

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        x = self.adap_pool(x)
        x = x.view(x.size(0), -1)  # 展平操作

        x = self.fc(x)
        x = self.tanh(x)  # 触发 Tanh 限制
        return x