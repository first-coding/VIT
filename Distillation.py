import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ====== 教师模型导入 ======
from models.Models import VisionTransformer  # 确保路径正确

# ====== 学生模型定义 ======
class StudentCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentCNN, self).__init__()
        self.features = nn.Sequential(
            self._make_block(3, 64),    # ConvBlock1
            self._make_block(64, 128),  # ConvBlock2
            self._make_block(128, 256), # ConvBlock3
            nn.AdaptiveAvgPool2d((1, 1))  # 全局池化
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),  # 隐藏层
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def _make_block(self, in_channels, out_channels):
        # 一个卷积块，包含两个卷积层 + BatchNorm + ReLU
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

# ====== 蒸馏损失函数（输出层）======
def kd_loss(student_logits, teacher_logits, T=6.0, alpha=0.6):
    ce_loss = F.cross_entropy(student_logits, teacher_logits.argmax(dim=1))
    kl_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    return alpha * ce_loss + (1 - alpha) * kl_loss

# ====== 数据加载 ======
def get_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    train_set = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ====== 测试函数 ======
def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    return acc

# ====== 主函数 ======
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_dataloader(batch_size=64)

    teacher = VisionTransformer()
    teacher.load_state_dict(torch.load('./models/vit_model_cycle_10.pth', map_location=device))
    teacher.to(device)
    teacher.eval()

    student = StudentCNN(num_classes=10).to(device)
    optimizer = optim.Adam(student.parameters(), lr=3e-4)

    epochs = 20
    T = 6.0
    alpha = 0.6

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, _ in loop:
            imgs = imgs.to(device)

            with torch.no_grad():
                teacher_logits = teacher(imgs)

            student_logits = student(imgs)
            loss = kd_loss(student_logits, teacher_logits, T=T, alpha=alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        acc = evaluate(student, test_loader, device)
        print(f"✅ Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader):.4f} | 🎯 Test Acc: {acc*100:.2f}%")

    torch.save(student.state_dict(), "student_cnn.pth")
    print("✅ 学生模型已保存为 student_cnn.pth")

if __name__ == '__main__':
    main()
