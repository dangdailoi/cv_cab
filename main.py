import os
from fastapi import FastAPI, File, UploadFile
import torch
import cv2
import numpy as np
from PIL import Image
import io
from io import BytesIO
from torchvision import transforms
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# Khởi tạo FastAPI app
app = FastAPI()

# Load YOLOv5 model
from ultralytics import YOLO
yolo_model = YOLO('model/best.pt')

# Load EfficientNet-B0 model
efficientnet_model = torch.load('model/efficientnet_b0.pth', map_location='cpu')
efficientnet_model.eval()

# Image preprocessing for EfficientNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# Labels
fault_labels = {0: 'broken strand', 1: 'welded strand', 2: 'bent strand', 3: 'long scratch', 4: 'crushed', 5: 'spaced strand', 6: 'deposit'}
severity_labels = {0: 'light', 1: 'deep', 2: 'important', 3: 'partial', 4: 'complete', 5: 'extracted', 6: 'superficial'}

# Hàm xử lý ảnh
def process_image(image_bytes):
    # Chuyển đổi byte ảnh thành ảnh OpenCV
    img = Image.open(BytesIO(image_bytes))
    img_rgb = np.array(img.convert('RGB'))

    # YOLO detect
    results = yolo_model(img_rgb)[0]

    annotated_img = img_rgb.copy()
    output_info = []

    for box in results.boxes:
        # BBox info
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.3:
            continue
        # Crop region
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:  # Phòng trường hợp crop lỗi
            continue

        crop_pil = Image.fromarray(crop)
        crop_tensor = transform(crop_pil).unsqueeze(0)

        # Predict severity
        with torch.no_grad():
            severity_logits = efficientnet_model(crop_tensor)
            severity = torch.argmax(severity_logits, dim=1).item()

        # Draw bounding box + labels
        label_text = f"{fault_labels.get(cls_id, 'Unknown')} ({severity_labels.get(severity, 'Unknown')}) {conf:.2f}"
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated_img, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Save info
        output_info.append({
            'bbox': [x1, y1, x2, y2],
            'fault_type': fault_labels.get(cls_id, 'Unknown'),
            'severity': severity_labels.get(severity, 'Unknown'),
            'confidence': conf
        })

    # Convert annotated image to byte stream to return
    _, img_encoded = cv2.imencode('.jpg', annotated_img)
    img_bytes = img_encoded.tobytes()

    return img_bytes, output_info

# API endpoint nhận ảnh
@app.post("/process/")
async def process_image_endpoint(file: UploadFile = File(...)):
    # Đọc ảnh từ file upload
    image_bytes = await file.read()

    # Xử lý ảnh
    annotated_img, output_info = process_image(image_bytes)

    # Trả ảnh đã được đánh dấu và thông tin dưới dạng JSON
    return JSONResponse(content={"image": annotated_img, "info": output_info})

# Endpoint gốc để kiểm tra
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}

# Cấu hình để chạy trên cổng mà Render cung cấp
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Lấy cổng từ biến môi trường
    uvicorn.run(app, host="0.0.0.0", port=port)
