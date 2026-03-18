import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioHashNet(nn.Module):
    def __init__(self, hash_bits=128):
        super(AudioHashNet, self).__init__()
        # 1. 卷积层：像识别图片一样提取频谱图的纹理
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # 2. 全连接层：将特征降维
        self.fc1 = nn.Linear(64 * 32 * 53, 512)  # 这里的尺寸需要根据输入频谱微调

        # 3. 关键：哈希映射层（Hash Layer）
        self.hash_layer = nn.Linear(512, hash_bits)

    def forward(self, x):
        # x 形状: [batch, 1, 128, 216] (128是梅尔数，216是时间帧)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))

        # 使用 Tanh 将输出挤压到 -1 到 1 之间
        feature = torch.tanh(self.hash_layer(x))
        return feature


# 测试一下模型能不能跑通
if __name__ == "__main__":
    model = AudioHashNet(hash_bits=128)
    # 模拟一个随机的频谱图输入
    test_input = torch.randn(1, 1, 128, 216)
    output = model(test_input)
    print("生成的特征维度:", output.shape)  # 应该是 [1, 128]

    # 变成二进制码：大于0设为1，小于0设为0
    binary_code = (output > 0).int()
    print("生成的二进制指纹片段:", binary_code[0][:10])
