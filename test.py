from os import path
import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.Models import VisionTransformer
import os
def main():
    # 1. 定义数据变换
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ColorJitter(brightness=(0.8, 1.2), contrast=0.0, saturation=0.0, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 2. 加载模型
    for i in range(1,11):
        model = VisionTransformer()
        origin_path = './AI_DataAnalysis/Vision Transformer/'
        paths = './vit_model_cycle_' + str(i) + '.pth'
        full_path = os.path.join(origin_path, paths)
        state_dict = torch.load(full_path)
        print(paths)
        model.load_state_dict(state_dict)  # 将状态字典加载到模型中
        model.eval()

        # 3. 选择设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # 4. 加载测试数据集
        test_dataset = datasets.CIFAR10(root='./AI_DataAnalysis/Vision Transformer/data', train=False, download=False, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

        # 5. 评估模型
        correct = 0
        total = 0

        with torch.no_grad():
            with tqdm.tqdm(total=len(test_loader), desc="Evaluating", leave=False) as pbar:
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    pbar.update(1)

        # 6. 计算准确率
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
