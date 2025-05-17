FROM python:3.9.21

WORKDIR /app

# 拷贝依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# 拷贝项目代码
COPY . .

EXPOSE 8000

# 环境变量设置
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
