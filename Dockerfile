# Sử dụng một base image Python nhẹ
FROM python:3.10-slim

# Đặt thư mục làm việc bên trong container
WORKDIR /app

# Copy file requirements trước để tận dụng cache của Docker
COPY requirements.txt .

# Cài đặt các thư viện
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code của project vào container
COPY . .

# Mở port 8501 của Streamlit
EXPOSE 8501

# Lệnh để chạy ứng dụng khi container khởi động
CMD ["streamlit", "run", "app.py"]