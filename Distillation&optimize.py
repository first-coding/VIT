import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
# ====== 教师模型导入 ======
from models.teacher_VIT import VisionTransformer  # 确保路径正确
from models.Student import StudentCNN  
from script.quantize_export_infer import quantize
# ====== 蒸馏损失函数（输出层）======
def kd_loss(student_logits, teacher_logits, T=6.0, alpha=0.6):
    # 计算交叉熵损失
    ce_loss = F.cross_entropy(student_logits, teacher_logits.argmax(dim=1))
    
    # 计算KL散度损失
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
        transforms.RandomHorizontalFlip(),  # 数据增强
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
    teacher.load_state_dict(torch.load('./Output/teacher/vit_model_cycle_10.pth', map_location=device))
    teacher.to(device)
    teacher.eval()

    student = StudentCNN(num_classes=10).to(device)
    optimizer = optim.Adam(student.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    epochs = 20
    T = 6.0
    alpha = 0.6
    best_acc = 0.0  # 🟡 初始化最优精度

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

        scheduler.step()
        acc = evaluate(student, test_loader, device)
        print(f"✅ Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader):.4f} | 🎯 Test Acc: {acc*100:.2f}%")

        # 🟢 若当前精度优于最佳，则保存模型
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), "./Output/student/best_student_cnn.pth")
            print(f"📦 保存新最优模型（精度: {best_acc*100:.2f}%）")

    print("🏁 训练完成。最优模型已保存为 best_student_cnn.pth")
    print("🚀 开始执行静态量化 + ONNX 导出 + 推理...")
    quantize()


if __name__ == '__main__':
    main()
