import numpy as np
from torchvision import transforms

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)  # Add batch dim
    return tensor.numpy().astype(np.float32)
