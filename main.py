import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# 定义数据集根目录和转换
root_dir = '/data/'
transform = transforms.ToTensor()  # 将数据转换为张量格式

# 加载训练集和测试集
train_dataset = dsets.CIFAR10(root=root_dir, train=True, transform=transform, download=False)
test_dataset = dsets.CIFAR10(root=root_dir, train=False, transform=transform, download=False)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 示例：遍历训练集的一个批次数据
for images, labels in train_loader:
    # images 是一个形状为 (batch_size, channels, height, width) 的张量
    # labels 是一个形状为 (batch_size,) 的张量，包含每张图片对应的标签
    # 在这里你可以对数据进行你想要的操作，比如训练你的模型等
    print(images.shape, labels)
