# Dùng Python image chính thức
FROM python:3.10-slim

# Tạo thư mục /app bên trong container
WORKDIR /app

# Copy toàn bộ mã nguồn (trong thư mục hiện tại) vào thư mục /app
COPY . /app

# Cài đặt các thư viện yêu cầu
RUN pip install --no-cache-dir -r requirements.txt

# Mở port 8000
EXPOSE 8000

# Chạy FastAPI app với uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
