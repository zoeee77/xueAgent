FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# 暴露端口
EXPOSE 8501

# 启动 Streamlit（后端通过 http 调用本地的 localhost:8000）
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.headless=true"]
