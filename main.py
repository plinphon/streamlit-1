import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page config
st.set_page_config(page_title="YOLO Image Detection", page_icon="🖼️")

st.title("🖼️ YOLO Image Detection App")
st.write("Upload an image and run YOLO detection")

# Load YOLO model (cached)
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # or your custom model: best.pt
    return model

# Load model
try:
    model = load_model()
    st.success("✅ YOLO model loaded")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Run YOLO"):
        with st.spinner("Running YOLO..."):
            # YOLO inference
            results = model(image)

            # Render result image
            annotated = results[0].plot()
            st.image(annotated, caption="Detection Result", use_container_width=True)

            # Show detections
            st.subheader("Detections")
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                st.write(f"- **{label}** ({conf*100:.2f}%)")
