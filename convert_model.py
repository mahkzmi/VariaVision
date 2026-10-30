from ultralytics import YOLO

# لود کردن مدل اصلی
model = YOLO("best.pt")

# تبدیل به ONNX (فرمت سبک‌تر و بهینه‌تر برای CPU)
model.export(format="onnx", imgsz=640, simplify=True)

print("✅ best.onnx created")