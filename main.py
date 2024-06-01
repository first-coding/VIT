import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import config
from models.Models import VisionTransformer

def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(root='/data', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    test_dataset = datasets.CIFAR10(root='/data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = VisionTransformer().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    train_losses = []
    test_accuracies = []

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} - Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch [{epoch+1}/{config["epochs"]}], Train Loss: {avg_train_loss:.4f}')

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{config['epochs']} - Evaluating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f'Epoch [{epoch+1}/{config["epochs"]}], Test Accuracy: {accuracy:.2f}%')

    # Save the final model
    torch.save(model.state_dict(), 'vit_model.pth')

    # Plotting
    epochs = range(1, config['epochs'] + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)    
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, 'b', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_evaluation_metrics.png')
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()
