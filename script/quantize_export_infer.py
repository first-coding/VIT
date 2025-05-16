import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

from models.Student import StudentCNN  # 修改为你的学生模型路径

# ====== 数据加载 ======
def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    calib_loader = DataLoader(datasets.CIFAR10(root='./data', train=True, download=False, transform=transform), batch_size=32, shuffle=False)
    test_loader = DataLoader(datasets.CIFAR10(root='./data', train=False, download=False, transform=transform), batch_size=32, shuffle=False)
    return calib_loader, test_loader

# ====== 精度评估 ======
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            out = model(imgs)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ====== 导出 ONNX ======
def export_onnx(model, path="./Output/Onnx/student_cnn.onnx"):
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX 模型已导出并验证成功：{path}")

# ====== ONNXRuntime 推理 ======
def run_onnx_inference(onnx_path, dataloader):
    # Session Options 配置
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4         # 单个操作使用的线程数
    so.inter_op_num_threads = 2         # 并行操作间使用的线程数
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # 启用所有图优化

    # 使用 CUDA 或 CPU 执行
    providers = ['CUDAExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    session = ort.InferenceSession("Output/Onnx/student_cnn.onnx", sess_options=so, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    correct, total = 0, 0
    for images, labels in dataloader:
        images_np = images.numpy().astype(np.float32)
        outputs = session.run([output_name], {input_name: images_np})[0]
        preds = np.argmax(outputs, axis=1)
        correct += (preds == labels.numpy()).sum()
        total += labels.size(0)

    acc = correct / total
    print(f"✅ ONNX 推理精度: {acc * 100:.2f}%")

# ====== 数据预处理类 ======
class CIFAR10DataReader(CalibrationDataReader):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)

    def get_next(self):
        try:
            images, _ = next(self.data_iter)
            return {'input': images.numpy()}
        except StopIteration:
            return None

# ====== 主流程 ======
def quantize():
    print("🚀 加载模型...")
    model = StudentCNN(num_classes=10)
    model.load_state_dict(torch.load("Output/student/best_student_cnn.pth", map_location='cpu'))
    model.eval()

    calib_loader, test_loader = get_dataloaders()

    print("📦 导出 ONNX 模型...")
    export_onnx(model)

    print("📏 开始使用 ONNXRuntime 进行静态量化...")
    calib_reader = CIFAR10DataReader(calib_loader)
    
    # 执行量化
    quantize_static(
        model_input="./Output/Onnx/student_cnn.onnx",
        model_output="./Output/Onnx/student_cnn_int8.onnx",
        calibration_data_reader=calib_reader,
        quant_format="QOperator",   # 使用 QOperator 格式
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8
    )
    print("🎯 ONNX 模型量化完成！")

    print("🎯 量化模型精度测试...")
    run_onnx_inference("./Output/Onnx/student_cnn_int8.onnx", test_loader)
    print("🏁 量化 + 导出 + 推理 完成！")
