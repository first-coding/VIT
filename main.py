import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from config import config
from models.Models import VisionTransformer

def get_device():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f'Available GPUs: {gpu_count}')
        for i in range(gpu_count):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        selected_gpu_index = 0
        device = torch.device(f'cuda:{selected_gpu_index}')
        device_name = torch.cuda.get_device_name(selected_gpu_index)
    else:
        device = torch.device('cpu')
        device_name = 'CPU'
    return device, device_name

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_and_evaluate(device, device_name):
    transform_train = transforms.Compose([
        transforms.RandomCrop(config['img_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.8,1.2), contrast=0.0, saturation=0.0, hue=0.01),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomVerticalFlip(),  # 新增：随机垂直翻转
        transforms.RandomResizedCrop(size=config['img_size']),  # 新增：随机裁剪
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ColorJitter(brightness=(0.8,1.2), contrast=0.0, saturation=0.0, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./AI_DataAnalysis/Vision Transformer/data', train=True, download=False, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)

    test_dataset = datasets.CIFAR10(root='./AI_DataAnalysis/Vision Transformer/data', train=False, download=False, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)

    model = VisionTransformer().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    optimizer = config['optimizer'](model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = config['scheduler'](optimizer, T_max=config['epochs'])

    train_losses = []
    test_accuracies = []

    total_epochs = config['epochs']
    cycle_length = 10

    alpha_values = [0.5, 1.0, 1.5]  # 新增：尝试不同的 alpha 值

    scaler = torch.cuda.amp.GradScaler()  # 使用混合精度训练

    for cycle in range(cycle_length):
        print(f'Starting cycle {cycle + 1}/{cycle_length}')
        for epoch in range(total_epochs):
            actual_epoch = cycle * cycle_length + epoch + 1
            model.train()
            train_loss = 0
            correct = 0
            total = 0

            # 新增：随机选择 alpha 值
            alpha = np.random.choice(alpha_values)
            with tqdm(total=len(train_loader), desc=f"Cycle {cycle + 1}/{cycle_length} - Epoch {epoch + 1}/{total_epochs} - Training", leave=False) as pbar:
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)

                    # MixUp data augmentation
                    images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=alpha)

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += loss.item()

                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    pbar.set_postfix(loss=loss.item(), accuracy=100. * correct / total)
                    pbar.update(1)

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f'Epoch [{actual_epoch}/{total_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {100. * correct / total:.2f}%')

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                with tqdm(total=len(test_loader), desc=f"Cycle {cycle + 1}/{cycle_length} - Epoch {epoch + 1}/{total_epochs} - Evaluating", leave=False) as pbar:
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        pbar.update(1)
            accuracy = 100 * correct / total
            test_accuracies.append(accuracy)
            print(f'Epoch [{actual_epoch}/{total_epochs}], Test Accuracy: {accuracy:.2f}%')

            scheduler.step()

        torch.save(model.state_dict(), f'vit_model_cycle_{cycle + 1}.pth')
        print(f'Model saved after cycle {cycle + 1}/10')

    torch.save(model.state_dict(), 'vit_model_final.pth')
    epochs = range(1, total_epochs + 1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'r', label='Training loss')
    plt.plot(epochs, test_accuracies, 'b', label='Test Accuracy')
    plt.title('Training Loss and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    lr_values = [round(scheduler.get_last_lr()[0], 4) for _ in range(total_epochs)]
    plt.plot(epochs, lr_values, 'g', label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_accuracies, 'b', label='Test Accuracy')
    plt.title('Test Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

if __name__ == "__main__":
    device, device_name = get_device()
    print(f'Using device: {device_name}')
    train_and_evaluate(device, device_name)
