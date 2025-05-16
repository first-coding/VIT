from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import uvicorn
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import torchvision.transforms as transforms
import os

# CIFAR-10 类别标签（全局变量）
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and template setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Model Setup
so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.inter_op_num_threads = 2
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
providers = ["CUDAExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
session = ort.InferenceSession("Output/Onnx/student_cnn_int8.onnx", sess_options=so, providers=providers)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Transform definition
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

def preprocess_image(image: Image.Image):
    image = transform(image).unsqueeze(0)
    return image.numpy().astype(np.float32)

# Home route: upload page
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocess_image(image)

        # Run inference
        outputs = session.run([output_name], {input_name: input_tensor})[0]
        pred_class = int(np.argmax(outputs))  # 获取预测的类别索引
        pred_label = class_names[pred_class]  # 获取类别名称

        # Return only class label in response
        return JSONResponse({
            "filename": file.filename,
            "prediction": pred_label  # 返回的是类别名称
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
