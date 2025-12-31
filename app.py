import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# 🚀 VisionGuard-AI Setup
# -----------------------------
st.set_page_config(page_title="VisionGuard-AI", layout="wide", page_icon="🛡️")
st.markdown("<h2 align='center'>🛡️ VisionGuard-AI</h2>", unsafe_allow_html=True)
st.markdown("<h5 align='center'>Object Detection + Confidence Analysis + Graph</h5>", unsafe_allow_html=True)

# -----------------------------
# 📁 Model Setup
# -----------------------------
MODEL_PATH = "models/yolov8n.pt"
RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model yolov8n.pt not found! Place it into the /models folder.")
    st.stop()

model = YOLO(MODEL_PATH)
st.success("🚀 Model Loaded!")

# -----------------------------
# 🛠️ Controls
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.4)
with col2:
    show_graph = st.checkbox("📊 Show Confidence Bar Graph", True)

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📌 Uploaded Image", width=600)

    # Detection
    results = model.predict(image, conf=conf_threshold)
    boxes = results[0].boxes

    # Save output
    result_image_path = os.path.join(RESULT_DIR, "detected_output.jpg")
    results[0].save(filename=result_image_path)
    st.image(result_image_path, caption="🎯 Detection Output", width=600)

    # -----------------------------
    # 📊 Detection Summary
    # -----------------------------
    if len(boxes) > 0:
        labels = [model.names[int(b.cls[0])] for b in boxes]
        confidences = [float(b.conf[0]) for b in boxes]

        df = pd.DataFrame({
            "Detected Object": labels,
            "Confidence": [round(c, 3) for c in confidences]
        })

        st.subheader("📌 Detection Summary Table")
        st.dataframe(df, use_container_width=True)
        st.success(f"🟢 Total Objects Detected: **{len(boxes)}**")

        # -----------------------------
        # 📈 Bar Graph (Confidence)
        # -----------------------------
        if show_graph:
            st.subheader("📊 Confidence Bar Graph")

            # Get highest confidence
            max_idx = confidences.index(max(confidences))
            top_label = labels[max_idx]
            top_score = confidences[max_idx]

            fig, ax = plt.subplots()
            bars = ax.bar(labels, confidences, color="skyblue")
            ax.set_xlabel("Detected Objects")
            ax.set_ylabel("Confidence Score")
            ax.set_title("Object Confidence Comparison")

            for i, val in enumerate(confidences):
                ax.text(i, val + 0.02, f"{val:.2f}", ha='center')

            st.pyplot(fig)

            st.info(f"🏆 Highest Confidence Object: **{top_label}** ({top_score:.2f})")

    else:
        st.warning("⚠ No objects detected at this confidence threshold.")

    # -----------------------------
    # ⬇️ Download Result
    # -----------------------------
    with open(result_image_path, "rb") as file:
        st.download_button(
            label="⬇️ Download Result Image",
            data=file,
            file_name="VisionGuardAI_output.jpg",
            mime="image/jpeg"
        )
