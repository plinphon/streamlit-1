from ultralytics import YOLO

def load_model():
    model = YOLO("yolov8n.pt")  # or your custom model: best.pt
    return model

model = load_model
