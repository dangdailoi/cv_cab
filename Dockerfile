# Dùng base image Python
FROM python:3.10-slim

# Cài thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy mã nguồn vào Docker image
COPY . /app

# Cài đặt Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Mở port 8000
EXPOSE 8000

# Lệnh để chạy app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
