import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "best_mobilenetv2_drowsiness.keras"
best_model = tf.keras.models.load_model(MODEL_PATH)

# Class index contract from training pipeline.
IMG_SIZE = 160
CLASS_NAMES = {
    0: "yawn",
    1: "no_yawn",
    2: "eyes_closed",
    3: "eyes_open",
}

EYE_CLASS_IDS = [2, 3]
MOUTH_CLASS_IDS = [0, 1]
CONFIDENCE_THRESHOLD = 0.55

# Reuse the best checkpoint if available, otherwise fallback to in-memory model objects.
checkpoint_path = "best_mobilenetv2_drowsiness.keras"
if os.path.exists(checkpoint_path):
    model_for_inference = load_model(checkpoint_path)
elif "best_model" in globals():
    model_for_inference = best_model
else:
    model_for_inference = best_model

face_cascade_infer = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade_infer = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def preprocess_roi_for_model(roi_bgr, img_size=IMG_SIZE):
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    resized = cv2.resize(roi_bgr, (img_size, img_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = preprocess_input(rgb)
    return np.expand_dims(x, axis=0)

def predict_subset(roi_bgr, allowed_ids):
    x = preprocess_roi_for_model(roi_bgr)
    if x is None:
        return "not_detected", 0.0
    probs = model_for_inference.predict(x, verbose=0)[0]
    subset = np.array([probs[i] for i in allowed_ids], dtype=np.float32)
    subset = subset / (subset.sum() + 1e-8)
    best_local_idx = int(np.argmax(subset))
    class_id = allowed_ids[best_local_idx]
    confidence = float(subset[best_local_idx])
    if confidence < CONFIDENCE_THRESHOLD:
        return "uncertain", confidence
    return CLASS_NAMES[class_id], confidence

def draw_box_with_label(image_bgr, box, label, confidence, color):
    x, y, w, h = box
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
    text = f"{label} ({confidence:.2f})"
    cv2.putText(image_bgr, text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

def mouth_box_from_face(face_box):
    x, y, w, h = face_box
    mx = x + int(0.18 * w)
    my = y + int(0.58 * h)
    mw = int(0.64 * w)
    mh = int(0.34 * h)
    return (mx, my, mw, mh)

def detect_and_annotate_states(image_bgr):
    output = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_infer.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        # Fallback: run mouth-state prediction on the full frame.
        label, conf = predict_subset(image_bgr, MOUTH_CLASS_IDS)
        h, w = image_bgr.shape[:2]
        draw_box_with_label(output, (10, 10, w - 20, h - 20), f"fallback_mouth:{label}", conf, (0, 165, 255))
        return output

    for (x, y, w, h) in faces:
        face_box = (x, y, w, h)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(output, "face", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)

        face_gray = gray[y:y+h, x:x+w]
        face_bgr = image_bgr[y:y+h, x:x+w]

        eyes = eye_cascade_infer.detectMultiScale(face_gray, scaleFactor=1.15, minNeighbors=6, minSize=(20, 20))
        eyes = sorted(eyes, key=lambda e: e[0])[:2]

        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_bgr[ey:ey+eh, ex:ex+ew]
            eye_label, eye_conf = predict_subset(eye_roi, EYE_CLASS_IDS)
            global_eye_box = (x + ex, y + ey, ew, eh)
            draw_box_with_label(output, global_eye_box, f"eye:{eye_label}", eye_conf, (0, 255, 0))

        mbox = mouth_box_from_face(face_box)
        mx, my, mw, mh = mbox
        mouth_roi = image_bgr[my:my+mh, mx:mx+mw]
        mouth_label, mouth_conf = predict_subset(mouth_roi, MOUTH_CLASS_IDS)
        draw_box_with_label(output, mbox, f"mouth:{mouth_label}", mouth_conf, (0, 0, 255))

    return output

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Make sure camera permissions are enabled.")

    print("Webcam started. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated = detect_and_annotate_states(frame)
        cv2.imshow("Drowsiness Demo - face/eyes/mouth", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stopped.")

# Execution
if __name__ == "__main__":
    main()