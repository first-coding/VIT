# Vision Transformer + Knowledge Distillation + Onnx Deployment

### Paper Reference
This project is based on the Vision Transformer (ViT) paper:
https://arxiv.org/abs/2010.11929

### Project Overview
This repository implements an end-to-end image classification pipeline that includes:

- Training a Vision Transformer (ViT) on CIFAR-10 as a teacher model
- Applying offline knowledge distillation to train a lightweight CNN student model
- Exporting the student model to ONNX format and applying static INT8 quantization
- Deploying fast inference using ONNX Runtime with multithreading and GPU acceleration
- Building a FastAPI HTTP service for prediction via image file uploads
- Packaging the application with Docker for easy deployment and reproducibility



## Getting Started
### Docker
1.Build Docker image:
```bash
docker build -t my-fastapi-app .
```
2.Run Container
```bash
docker run -p 12345:8000 my-fastapi-app
```
Access the API at: http://localhost:12345


### Without Docker
```
pip install -r requirements.txt
```
```
python main.py
```

## Train
Modify hyperparameters in Config/config/ as needed.
- Train the teacher ViT model:
``` bash
python teacher_model_train.py
```
- Evaluate the trained teacher:
```bash
python teacher_model_test.py
```
- Perform knowledge distillation + optimize
```bash
python Distillation&optimize.py
```
### Dataset
The data can be downloaded from the official website or in the image place, or the Download setting to True when the code loads the data


## Knowledge Distillation
This project uses offline knowledge distillation to compress a Vision Transformer into a lightweight CNN model.

### Why?
While ViTs achieve high accuracy, they are too heavy for edge deployment. Distillation transfers soft label knowledge from a large model to a small one, preserving performance while reducing computational cost.

### Core Concepts
**Hard Labels**: Ground-truth one-hot targets\
**Soft Labels**: Teacher model output (logits), softened with temperature T
The total loss is:
```bash
Loss = α * KD_loss + (1 - α) * CE_loss
```
### Where:
- **KD_loss**: KL divergence between teacher and student logits
- **CE_loss**: Cross-entropy between student prediction and labels
- **α**: balance weight
- **T**: temperature hyperparameter\
See Distillation&optimize.py for implementation.

### Directory Structure 
```
├── Config/                   # Configuration files
├── model/                    # ViT and CNN model definitions
├── script/                   # ONNX export, quantization, and inference scripts
├── teacher_model_train.py    # ViT training entry
├── teacher_model_test.py     # ViT evaluation
├── Distillation&optimize.py  # Student model distillation and optimization
├── main.py                   # FastAPI inference service
├── Dockerfile                # Container build
├── requirements.txt          # Python dependencies
└── README.md
```
## Performance & Contributing
Due to time and resource limitations, the ViT teacher model achieved 74% accuracy on the CIFAR-10 test set, while the distilled CNN student model reached 64%. Inference was further optimized using ONNX Runtime with INT8 quantization for faster and more efficient deployment.

If you manage to improve the model architecture, training strategy, or deployment pipeline, feel free to open a pull request or start a discussion — contributions and suggestions are always welcome!
