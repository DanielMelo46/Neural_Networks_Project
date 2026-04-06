import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗", layout="centered")

st.title("🚗 Driver Drowsiness Detection System")
st.write("Upload an image to detect whether the driver is drowsy or alert.")

class_names = {
    0: "yawn",
    1: "no_yawn",
    2: "eyes_closed",
    3: "eyes_open"
}

drowsy_classes = {"yawn", "eyes_closed"}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_mobilenetv2_drowsiness.keras")

try:
    model = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()

def prepare_image(image, target_size=(160, 160)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image):
    processed = prepare_image(image)
    preds = model.predict(processed, verbose=0)[0]

    pred_index = int(np.argmax(preds))
    pred_label = class_names[pred_index]
    confidence = float(preds[pred_index])
    status = "Drowsy" if pred_label in drowsy_classes else "Alert"

    return pred_label, confidence, preds, status

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Run Detection"):
        pred_label, confidence, preds, status = predict_image(image)

        st.subheader("Prediction Result")
        st.write(f"**Predicted Class:** {pred_label}")
        st.write(f"**Confidence:** {confidence:.2%}")
        st.write(f"**Driver Status:** {status}")

        if status == "Drowsy":
            st.error("⚠️ Warning: Drowsiness detected. Please take a break.")
        else:
            st.success("✅ Driver appears alert.")

        st.subheader("Class Probabilities")
        for i, prob in enumerate(preds):
            st.write(f"**{class_names[i]}:** {prob:.2%}")