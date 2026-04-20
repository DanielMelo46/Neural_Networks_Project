import threading
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗", layout="centered")

st.title("🚗 Driver Drowsiness Detection System")
st.write("Run live webcam inference and see what the model is detecting each frame.")

CLASS_NAMES = {
    0: "yawn",
    1: "no_yawn",
    2: "eyes_closed",
    3: "eyes_open",
}
EYE_CLASS_IDS = [2, 3]
MOUTH_CLASS_IDS = [0, 1]
CONFIDENCE_THRESHOLD = 0.55
IMG_SIZE = 160


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "best_mobilenetv2_drowsiness.keras"
    return tf.keras.models.load_model(model_path)


try:
    model = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()


def mouth_box_from_face(face_box):
    x, y, w, h = face_box
    mx = x + int(0.18 * w)
    my = y + int(0.58 * h)
    mw = int(0.64 * w)
    mh = int(0.34 * h)
    return (mx, my, mw, mh)


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

    probs = model.predict(x, verbose=0)[0]
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
    cv2.putText(
        image_bgr,
        text,
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_detection_text = "Waiting for webcam frames..."
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )

    def _detect_and_annotate_states(self, image_bgr):
        output = image_bgr.copy()
        h, w, _ = image_bgr.shape

        rgb_in = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_in)

        if not results or not results.detections:
            label, conf = predict_subset(image_bgr, MOUTH_CLASS_IDS)
            draw_box_with_label(
                output,
                (10, 10, w - 20, h - 20),
                f"fallback_mouth:{label}",
                conf,
                (0, 165, 255),
            )
            summary = f"Mouth: {label} ({conf:.2f})"
            return output, summary

        # Use only the largest face for a stable single-driver status label.
        def area(det):
            b = det.location_data.relative_bounding_box
            return b.width * b.height

        detection = max(results.detections, key=area)
        bbox = detection.location_data.relative_bounding_box
        fx, fy, fw, fh = (
            int(bbox.xmin * w),
            int(bbox.ymin * h),
            int(bbox.width * w),
            int(bbox.height * h),
        )

        fx, fy = max(0, fx), max(0, fy)
        fw, fh = min(fw, w - fx), min(fh, h - fy)
        face_box = (fx, fy, fw, fh)

        cv2.rectangle(output, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)
        cv2.putText(
            output,
            "face",
            (fx, max(20, fy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        eye_zone_h = int(fh * 0.45)
        eye_roi = image_bgr[fy : fy + eye_zone_h, fx : fx + fw]
        eye_label, eye_conf = predict_subset(eye_roi, EYE_CLASS_IDS)
        draw_box_with_label(output, (fx, fy, fw, eye_zone_h), f"eyes:{eye_label}", eye_conf, (0, 255, 0))

        mx, my, mw, mh = mouth_box_from_face(face_box)
        mx, my = max(0, mx), max(0, my)
        mw, mh = min(mw, w - mx), min(mh, h - my)

        mouth_roi = image_bgr[my : my + mh, mx : mx + mw]
        mouth_label, mouth_conf = predict_subset(mouth_roi, MOUTH_CLASS_IDS)
        draw_box_with_label(output, (mx, my, mw, mh), f"mouth:{mouth_label}", mouth_conf, (0, 0, 255))

        summary = f"Eyes: {eye_label} ({eye_conf:.2f}) | Mouth: {mouth_label} ({mouth_conf:.2f})"
        return output, summary

    def recv(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")
        annotated, summary = self._detect_and_annotate_states(image_bgr)

        with self.lock:
            self.latest_detection_text = summary

        # Add a dedicated label strip below the webcam frame.
        h, w = annotated.shape[:2]
        strip_h = 36
        framed = np.zeros((h + strip_h, w, 3), dtype=np.uint8)
        framed[:h, :] = annotated
        cv2.rectangle(framed, (0, h), (w, h + strip_h), (24, 24, 24), -1)
        cv2.putText(
            framed,
            f"Model detection: {summary}",
            (10, h + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(framed, format="bgr24")

    def get_latest_detection_text(self):
        with self.lock:
            return self.latest_detection_text


st.subheader("Live Webcam")
st.caption("Allow camera access, then click Start. A live label is shown under the webcam feed with current model detections.")

webrtc_ctx = webrtc_streamer(
    key="drowsiness-webcam",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

