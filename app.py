import threading
from pathlib import Path
from typing import Dict

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --bg-top: #eff6f8;
        --bg-bottom: #d7e6ea;
        --panel: rgba(255, 255, 255, 0.88);
        --panel-border: rgba(18, 60, 72, 0.14);
        --text-main: #132a33;
        --text-muted: #3f5f69;
        --safe: #1f7a4f;
        --warn: #a96400;
        --risk: #a22222;
        --idle: #526a73;
        --monitor: #2f6387;
    }

    .stApp {
        background: radial-gradient(circle at top right, #f8fbfc 0%, var(--bg-top) 42%, var(--bg-bottom) 100%);
        color: var(--text-main);
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.2px;
    }

    body, p, div, label, span {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #183642 0%, #214a59 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f2f7f9;
    }

    /* Keep dropdown controls readable on light input backgrounds. */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #121212 !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] span,
    [data-testid="stSidebar"] div[data-baseweb="select"] div {
        color: #121212 !important;
        -webkit-text-fill-color: #121212 !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] input {
        color: #121212 !important;
        -webkit-text-fill-color: #121212 !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] svg {
        fill: #121212 !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="input"] > div,
    [data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"] > div,
    [data-testid="stSidebar"] .stTextInput div[data-baseweb="input"] > div {
        background: #ffffff !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="input"] input,
    [data-testid="stSidebar"] .stNumberInput input,
    [data-testid="stSidebar"] .stTextInput input {
        color: #121212 !important;
        -webkit-text-fill-color: #121212 !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="input"] input::placeholder,
    [data-testid="stSidebar"] .stNumberInput input::placeholder,
    [data-testid="stSidebar"] .stTextInput input::placeholder {
        color: #4a4a4a !important;
        -webkit-text-fill-color: #4a4a4a !important;
        opacity: 1 !important;
    }

    div[role="listbox"] div,
    div[role="option"] {
        color: #121212 !important;
    }

    div[role="option"][aria-selected="true"] {
        color: #121212 !important;
        background: #e8edf1 !important;
    }

    .hero-card {
        padding: 1.0rem 1.25rem;
        border-radius: 16px;
        background: linear-gradient(120deg, rgba(255,255,255,0.92) 0%, rgba(255,255,255,0.76) 100%);
        border: 1px solid var(--panel-border);
        box-shadow: 0 12px 34px rgba(15, 41, 50, 0.12);
        margin-bottom: 0.65rem;
        animation: reveal 500ms ease;
    }

    .hero-kicker {
        text-transform: uppercase;
        font-size: 0.74rem;
        letter-spacing: 1.2px;
        color: #2b5663;
        margin-bottom: 0.25rem;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.25;
        margin: 0;
    }

    .hero-sub {
        color: var(--text-muted);
        margin-top: 0.38rem;
        font-size: 0.95rem;
    }

    .status-banner {
        border-radius: 14px;
        border: 1px solid var(--panel-border);
        border-left: 7px solid var(--monitor);
        background: var(--panel);
        padding: 0.85rem 1.0rem;
        box-shadow: 0 7px 22px rgba(20, 44, 58, 0.09);
        margin-bottom: 0.7rem;
        animation: reveal 450ms ease;
    }

    .status-label {
        color: var(--text-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        margin-bottom: 0.2rem;
    }

    .status-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
        margin-bottom: 0.22rem;
    }

    .status-sub {
        color: #2f515e;
        font-size: 0.9rem;
    }

    .metric-card {
        border-radius: 12px;
        border: 1px solid var(--panel-border);
        background: var(--panel);
        padding: 0.8rem 0.95rem;
        box-shadow: 0 6px 16px rgba(15, 40, 50, 0.08);
        min-height: 112px;
        animation: reveal 620ms ease;
    }

    .metric-label {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #45616b;
    }

    .metric-value {
        margin-top: 0.35rem;
        margin-bottom: 0.2rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.18rem;
        font-weight: 700;
        color: #16363f;
    }

    .metric-sub {
        color: #3f5e67;
        font-size: 0.86rem;
    }

    @keyframes reveal {
        from {
            opacity: 0;
            transform: translateY(6px);
        }
        to {
            opacity: 1;
            transform: translateY(0px);
        }
    }

    @media (max-width: 900px) {
        .hero-title {
            font-size: 1.35rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CLASS_NAMES = {
    0: "yawn",
    1: "no_yawn",
    2: "eyes_closed",
    3: "eyes_open",
}
EYE_CLASS_IDS = [2, 3]
MOUTH_CLASS_IDS = [0, 1]
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
DEFAULT_CLOSED_FRAMES = 5
DEFAULT_EMA_ALPHA = 0.35
IMG_SIZE = 160
DEFAULT_STREAM_WIDTH = 640
DEFAULT_STREAM_HEIGHT = 480
DEFAULT_STREAM_FPS = 15
DEFAULT_PROCESS_EVERY_N = 1
DEFAULT_DETECT_EVERY_N = 1
DEFAULT_UI_REFRESH_SECONDS = 2

PERFORMANCE_PRESETS = {
    "Balanced": {
        "stream_width": 640,
        "stream_height": 480,
        "stream_fps": 15,
        "process_every_n_frames": 1,
        "detect_every_n_frames": 1,
        "ui_refresh_seconds": 2,
    },
    "Fast": {
        "stream_width": 480,
        "stream_height": 360,
        "stream_fps": 12,
        "process_every_n_frames": 3,
        "detect_every_n_frames": 3,
        "ui_refresh_seconds": 3,
    },
}

STATUS_STYLE = {
    "Alert": {"hex": "#1f7a4f", "bgr": (79, 122, 31)},
    "Getting Drowsy": {"hex": "#a96400", "bgr": (0, 100, 169)},
    "Drowsy": {"hex": "#a22222", "bgr": (34, 34, 162)},
    "No Face Detected": {"hex": "#a22222", "bgr": (34, 34, 162)},
    "Idle": {"hex": "#526a73", "bgr": (115, 106, 82)},
    "Monitoring": {"hex": "#2f6387", "bgr": (135, 99, 47)},
}


def build_idle_metrics(
    message: str = "Waiting for webcam frames...",
    closed_frames_n: int = DEFAULT_CLOSED_FRAMES,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, object]:
    return {
        "driver_status": "Idle",
        "raw_eyes_label": "not_detected",
        "raw_eyes_conf": 0.0,
        "raw_mouth_label": "not_detected",
        "raw_mouth_conf": 0.0,
        "smoothed_eyes_label": "not_detected",
        "smoothed_eyes_conf": 0.0,
        "smoothed_mouth_label": "not_detected",
        "smoothed_mouth_conf": 0.0,
        "eye_closed_streak": 0,
        "closed_frames_n": int(closed_frames_n),
        "confidence_threshold": float(confidence_threshold),
        "summary": message,
    }


def init_session_state():
    defaults = {
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "closed_frame_threshold": DEFAULT_CLOSED_FRAMES,
        "ema_alpha": DEFAULT_EMA_ALPHA,
        "performance_profile": "Balanced",
        "applied_performance_profile": "",
        "stream_width": DEFAULT_STREAM_WIDTH,
        "stream_height": DEFAULT_STREAM_HEIGHT,
        "stream_fps": DEFAULT_STREAM_FPS,
        "process_every_n_frames": DEFAULT_PROCESS_EVERY_N,
        "detect_every_n_frames": DEFAULT_DETECT_EVERY_N,
        "ui_refresh_seconds": DEFAULT_UI_REFRESH_SECONDS,
        "was_playing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "best_mobilenetv2_drowsiness.keras"
    return tf.keras.models.load_model(model_path)


try:
    model = load_model()
    model_load_message = "Model loaded successfully."
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


def run_model_batch(batch):
    predictions = model(batch, training=False)
    if hasattr(predictions, "numpy"):
        predictions = predictions.numpy()
    return np.asarray(predictions, dtype=np.float32)


def classify_subset_probs(probs, allowed_ids, confidence_threshold):
    subset = np.array([probs[i] for i in allowed_ids], dtype=np.float32)
    subset = subset / (subset.sum() + 1e-8)
    label_probs = {CLASS_NAMES[allowed_ids[i]]: float(subset[i]) for i in range(len(allowed_ids))}

    best_local_idx = int(np.argmax(subset))
    class_id = allowed_ids[best_local_idx]
    confidence = float(subset[best_local_idx])

    if confidence < confidence_threshold:
        return "uncertain", confidence, label_probs
    return CLASS_NAMES[class_id], confidence, label_probs


def predict_subset(roi_bgr, allowed_ids, confidence_threshold):
    x = preprocess_roi_for_model(roi_bgr)
    if x is None:
        return "not_detected", 0.0, {}

    probs = run_model_batch(x)[0]
    return classify_subset_probs(probs, allowed_ids, confidence_threshold)


def predict_eye_and_mouth(eye_roi_bgr, mouth_roi_bgr, confidence_threshold):
    eye_x = preprocess_roi_for_model(eye_roi_bgr)
    mouth_x = preprocess_roi_for_model(mouth_roi_bgr)

    batched_inputs = []
    slot_names = []
    if eye_x is not None:
        batched_inputs.append(eye_x)
        slot_names.append("eye")
    if mouth_x is not None:
        batched_inputs.append(mouth_x)
        slot_names.append("mouth")

    eye_result = ("not_detected", 0.0, {})
    mouth_result = ("not_detected", 0.0, {})

    if not batched_inputs:
        return eye_result + mouth_result

    batch = np.concatenate(batched_inputs, axis=0)
    batch_probs = run_model_batch(batch)

    idx = 0
    for slot in slot_names:
        probs = batch_probs[idx]
        if slot == "eye":
            eye_result = classify_subset_probs(probs, EYE_CLASS_IDS, confidence_threshold)
        else:
            mouth_result = classify_subset_probs(probs, MOUTH_CLASS_IDS, confidence_threshold)
        idx += 1

    return eye_result + mouth_result


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
    def __init__(
        self,
        confidence_threshold,
        closed_frame_threshold,
        ema_alpha,
        process_every_n_frames,
        detect_every_n_frames,
    ):
        self.lock = threading.Lock()
        self.confidence_threshold = float(confidence_threshold)
        self.closed_frame_threshold = int(closed_frame_threshold)
        self.ema_alpha = float(ema_alpha)
        self.process_every_n_frames = max(1, int(process_every_n_frames))
        self.detect_every_n_frames = max(1, int(detect_every_n_frames))
        self.latest_detection_text = "Waiting for webcam frames..."
        self.latest_metrics = build_idle_metrics(
            closed_frames_n=self.closed_frame_threshold,
            confidence_threshold=self.confidence_threshold,
        )
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )
        self.smoothed_eye_closed_prob = None
        self.smoothed_yawn_prob = None
        self.eye_closed_streak = 0
        self.frame_count = 0
        self.last_face_box = None

    def _ema(self, previous, current):
        if previous is None:
            return float(current)
        return float(self.ema_alpha * current + (1.0 - self.ema_alpha) * previous)

    def _label_from_binary_prob(self, positive_prob, positive_label, negative_label):
        negative_prob = 1.0 - positive_prob
        if positive_prob >= self.confidence_threshold:
            return positive_label, float(positive_prob)
        if negative_prob >= self.confidence_threshold:
            return negative_label, float(negative_prob)
        return "uncertain", float(max(positive_prob, negative_prob))

    def update_runtime_config(
        self,
        confidence_threshold,
        closed_frame_threshold,
        ema_alpha,
        process_every_n_frames,
        detect_every_n_frames,
    ):
        with self.lock:
            self.confidence_threshold = float(confidence_threshold)
            self.closed_frame_threshold = int(closed_frame_threshold)
            self.ema_alpha = float(ema_alpha)
            self.process_every_n_frames = max(1, int(process_every_n_frames))
            self.detect_every_n_frames = max(1, int(detect_every_n_frames))

    def reset_state(self, message="Webcam stopped. Press Start to begin monitoring."):
        with self.lock:
            self.eye_closed_streak = 0
            self.smoothed_eye_closed_prob = None
            self.smoothed_yawn_prob = None
            self.frame_count = 0
            self.last_face_box = None
            self.latest_detection_text = message
            self.latest_metrics = build_idle_metrics(
                message=message,
                closed_frames_n=self.closed_frame_threshold,
                confidence_threshold=self.confidence_threshold,
            )

    def _resolve_face_box(self, image_bgr, run_face_detection):
        h, w, _ = image_bgr.shape

        if run_face_detection:
            rgb_in = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb_in)

            if results and results.detections:
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
                self.last_face_box = (fx, fy, fw, fh)
            else:
                self.last_face_box = None

        if self.last_face_box is None:
            return None

        fx, fy, fw, fh = self.last_face_box
        fx = max(0, min(fx, max(0, w - 2)))
        fy = max(0, min(fy, max(0, h - 2)))
        fw = min(fw, w - fx)
        fh = min(fh, h - fy)
        if fw < 2 or fh < 2:
            return None

        clipped = (fx, fy, fw, fh)
        self.last_face_box = clipped
        return clipped

    def _no_face_metrics(self, fallback_mouth_label, fallback_mouth_conf):
        self.eye_closed_streak = 0
        self.smoothed_eye_closed_prob = None
        self.smoothed_yawn_prob = None
        return {
            "driver_status": "No Face Detected",
            "raw_eyes_label": "not_detected",
            "raw_eyes_conf": 0.0,
            "raw_mouth_label": fallback_mouth_label,
            "raw_mouth_conf": float(fallback_mouth_conf),
            "smoothed_eyes_label": "not_detected",
            "smoothed_eyes_conf": 0.0,
            "smoothed_mouth_label": fallback_mouth_label,
            "smoothed_mouth_conf": float(fallback_mouth_conf),
            "eye_closed_streak": 0,
            "closed_frames_n": self.closed_frame_threshold,
            "confidence_threshold": self.confidence_threshold,
            "summary": f"ALERT: No Face Detected | This is abnormal | Mouth fallback: {fallback_mouth_label} ({fallback_mouth_conf:.2f})",
        }

    def _detect_and_annotate_states(self, image_bgr, run_face_detection=True):
        output = image_bgr.copy()
        h, w, _ = image_bgr.shape

        face_box = self._resolve_face_box(image_bgr, run_face_detection=run_face_detection)
        if face_box is None:
            fallback_label, fallback_conf, _ = predict_subset(
                image_bgr,
                MOUTH_CLASS_IDS,
                self.confidence_threshold,
            )
            cv2.rectangle(output, (10, 52), (max(12, w - 10), max(56, h - 10)), (0, 0, 255), 2)
            cv2.rectangle(output, (10, 10), (350, 46), (0, 0, 255), -1)
            cv2.putText(
                output,
                "ALERT: NO FACE DETECTED",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            metrics = self._no_face_metrics(fallback_label, fallback_conf)
            return output, metrics

        fx, fy, fw, fh = face_box
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

        eye_zone_h = max(2, int(fh * 0.45))
        eye_roi = image_bgr[fy : min(h, fy + eye_zone_h), fx : min(w, fx + fw)]

        mx, my, mw, mh = mouth_box_from_face(face_box)
        mx, my = max(0, mx), max(0, my)
        mw, mh = min(mw, w - mx), min(mh, h - my)
        mouth_roi = image_bgr[my : min(h, my + mh), mx : min(w, mx + mw)]
        (
            raw_eye_label,
            raw_eye_conf,
            eye_probs,
            raw_mouth_label,
            raw_mouth_conf,
            mouth_probs,
        ) = predict_eye_and_mouth(
            eye_roi,
            mouth_roi,
            self.confidence_threshold,
        )

        eye_closed_prob = float(eye_probs.get("eyes_closed", 0.0))
        yawn_prob = float(mouth_probs.get("yawn", 0.0))
        self.smoothed_eye_closed_prob = self._ema(self.smoothed_eye_closed_prob, eye_closed_prob)
        self.smoothed_yawn_prob = self._ema(self.smoothed_yawn_prob, yawn_prob)

        smoothed_eye_label, smoothed_eye_conf = self._label_from_binary_prob(
            self.smoothed_eye_closed_prob,
            "eyes_closed",
            "eyes_open",
        )
        smoothed_mouth_label, smoothed_mouth_conf = self._label_from_binary_prob(
            self.smoothed_yawn_prob,
            "yawn",
            "no_yawn",
        )

        if smoothed_eye_label == "eyes_closed":
            self.eye_closed_streak += 1
        else:
            self.eye_closed_streak = 0

        if self.eye_closed_streak >= self.closed_frame_threshold:
            driver_status = "Drowsy"
        elif smoothed_eye_label == "eyes_open" and smoothed_mouth_label == "yawn":
            driver_status = "Getting Drowsy"
        elif smoothed_eye_label == "eyes_open" and smoothed_mouth_label != "yawn":
            driver_status = "Alert"
        else:
            driver_status = "Monitoring"

        draw_box_with_label(
            output,
            (fx, fy, fw, eye_zone_h),
            f"eyes:{smoothed_eye_label}",
            smoothed_eye_conf,
            (0, 255, 0),
        )
        draw_box_with_label(
            output,
            (mx, my, max(2, mw), max(2, mh)),
            f"mouth:{smoothed_mouth_label}",
            smoothed_mouth_conf,
            (0, 0, 255),
        )

        status_color = STATUS_STYLE.get(driver_status, STATUS_STYLE["Monitoring"])["bgr"]
        if driver_status == "Drowsy":
            cv2.rectangle(output, (0, 0), (w - 1, h - 1), status_color, 4)
            cv2.rectangle(output, (10, 10), (350, 50), status_color, -1)
            cv2.putText(
                output,
                "ALERT: DROWSY DRIVER",
                (18, 37),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.rectangle(output, (10, 10), (260, 42), status_color, -1)
            cv2.putText(
                output,
                driver_status,
                (18, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        summary = (
            f"{driver_status} | Eyes: {smoothed_eye_label} ({smoothed_eye_conf:.2f}) | "
            f"Mouth: {smoothed_mouth_label} ({smoothed_mouth_conf:.2f}) | "
            f"Closed streak: {self.eye_closed_streak}/{self.closed_frame_threshold}"
        )
        metrics = {
            "driver_status": driver_status,
            "raw_eyes_label": raw_eye_label,
            "raw_eyes_conf": float(raw_eye_conf),
            "raw_mouth_label": raw_mouth_label,
            "raw_mouth_conf": float(raw_mouth_conf),
            "smoothed_eyes_label": smoothed_eye_label,
            "smoothed_eyes_conf": float(smoothed_eye_conf),
            "smoothed_mouth_label": smoothed_mouth_label,
            "smoothed_mouth_conf": float(smoothed_mouth_conf),
            "eye_closed_streak": int(self.eye_closed_streak),
            "closed_frames_n": int(self.closed_frame_threshold),
            "confidence_threshold": float(self.confidence_threshold),
            "summary": summary,
        }
        return output, metrics

    def recv(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        should_process_frame = self.frame_count == 1 or (self.frame_count % self.process_every_n_frames == 0)

        if should_process_frame:
            run_face_detection = self.last_face_box is None or (self.frame_count % self.detect_every_n_frames == 0)
            annotated, metrics = self._detect_and_annotate_states(image_bgr, run_face_detection=run_face_detection)
            summary = metrics["summary"]

            with self.lock:
                self.latest_detection_text = summary
                self.latest_metrics = metrics
        else:
            annotated = image_bgr.copy()
            with self.lock:
                summary = self.latest_detection_text
                metrics = dict(self.latest_metrics)

        status_style = STATUS_STYLE.get(metrics["driver_status"], STATUS_STYLE["Monitoring"])

        h, w = annotated.shape[:2]
        strip_h = 42
        framed = np.zeros((h + strip_h, w, 3), dtype=np.uint8)
        framed[:h, :] = annotated
        cv2.rectangle(framed, (0, h), (w, h + strip_h), (18, 24, 28), -1)
        cv2.rectangle(framed, (0, h), (8, h + strip_h), status_style["bgr"], -1)

        label_text = summary if len(summary) <= 118 else summary[:115] + "..."
        cv2.putText(
            framed,
            label_text,
            (14, h + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(framed, format="bgr24")

    def get_latest_detection_text(self):
        with self.lock:
            return self.latest_detection_text

    def get_latest_metrics(self):
        with self.lock:
            return dict(self.latest_metrics)


def status_hex(status):
    return STATUS_STYLE.get(status, STATUS_STYLE["Monitoring"])["hex"]


def render_live_cards(metrics):
    status = metrics.get("driver_status", "Idle")
    status_color = status_hex(status)
    st.markdown(
        f"""
        <div class="status-banner" style="border-left-color:{status_color};">
            <div class="status-label">Driver State</div>
            <div class="status-value" style="color:{status_color};">{status}</div>
            <div class="status-sub">{metrics.get("summary", "No summary available.")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4, gap="small")

    with cols[0]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Eyes (smoothed)</div>
                <div class="metric-value">{metrics.get("smoothed_eyes_label", "not_detected")}</div>
                <div class="metric-sub">confidence: {metrics.get("smoothed_eyes_conf", 0.0):.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[1]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Mouth (smoothed)</div>
                <div class="metric-value">{metrics.get("smoothed_mouth_label", "not_detected")}</div>
                <div class="metric-sub">confidence: {metrics.get("smoothed_mouth_conf", 0.0):.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[2]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Closed-eye streak</div>
                <div class="metric-value">{metrics.get("eye_closed_streak", 0)} / {metrics.get("closed_frames_n", 0)}</div>
                <div class="metric-sub">frames required for Drowsy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[3]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Threshold</div>
                <div class="metric-value">{metrics.get("confidence_threshold", 0.0):.2f}</div>
                <div class="metric-sub">raw eyes: {metrics.get("raw_eyes_label", "-")}, raw mouth: {metrics.get("raw_mouth_label", "-")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def get_display_metrics(webrtc_ctx):
    if bool(getattr(webrtc_ctx.state, "playing", False)) and webrtc_ctx.video_processor:
        return webrtc_ctx.video_processor.get_latest_metrics()

    idle_closed_frames_n = int(st.session_state.get("closed_frame_threshold", DEFAULT_CLOSED_FRAMES))
    idle_confidence_threshold = float(st.session_state.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))

    if webrtc_ctx.video_processor:
        return build_idle_metrics(
            message="Webcam is stopped. Press Start to resume monitoring.",
            closed_frames_n=idle_closed_frames_n,
            confidence_threshold=idle_confidence_threshold,
        )
    return build_idle_metrics(
        message="Allow camera access and press Start to begin.",
        closed_frames_n=idle_closed_frames_n,
        confidence_threshold=idle_confidence_threshold,
    )


def render_main_header():
    st.markdown(
        """
        <section class="hero-card">
            <div class="hero-kicker">Driver Safety Monitor</div>
            <h1 class="hero-title">Live Drowsiness Dashboard</h1>
            <div class="hero-sub">Face-based eye and mouth state predictions with smoothing and persistence-aware drowsiness logic.</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("### Performance")
    st.caption("Reduce browser lag by limiting stream load and processing cadence.")
    selected_profile = st.selectbox(
        "Performance profile",
        options=["Balanced", "Fast", "Custom"],
        key="performance_profile",
    )
    if selected_profile != st.session_state.get("applied_performance_profile"):
        preset = PERFORMANCE_PRESETS.get(selected_profile)
        if preset:
            for k, v in preset.items():
                st.session_state[k] = v
        st.session_state["applied_performance_profile"] = selected_profile

    st.slider("Stream width", min_value=320, max_value=1280, step=80, key="stream_width")
    st.slider("Stream height", min_value=240, max_value=720, step=40, key="stream_height")
    st.slider("Target FPS", min_value=8, max_value=30, step=1, key="stream_fps")
    st.number_input(
        "Inference every N frames",
        min_value=1,
        max_value=5,
        step=1,
        key="process_every_n_frames",
    )
    st.number_input(
        "Face detect every N frames",
        min_value=1,
        max_value=5,
        step=1,
        key="detect_every_n_frames",
    )
    st.slider(
        "Card refresh interval (seconds)",
        min_value=1,
        max_value=5,
        step=1,
        key="ui_refresh_seconds",
    )

    st.markdown("### Monitor Controls")
    st.caption("Tune confidence, smoothing, and persistence while the stream is running.")
    st.slider(
        "Confidence threshold",
        min_value=0.40,
        max_value=0.95,
        step=0.01,
        key="confidence_threshold",
    )
    st.number_input(
        "Closed-eye persistence N (frames)",
        min_value=3,
        max_value=40,
        step=1,
        key="closed_frame_threshold",
    )
    st.slider(
        "EMA smoothing alpha",
        min_value=0.10,
        max_value=0.90,
        step=0.05,
        key="ema_alpha",
    )

    st.markdown("### State Rules")
    st.caption("Alert: eyes_open + not yawn")
    st.caption("Getting Drowsy: eyes_open + yawn")
    st.caption("Drowsy: eyes_closed persists for N frames")
    st.success(model_load_message)


render_main_header()
st.subheader("Live Webcam")
st.caption("Allow camera access, then click Start. The webcam strip and cards reflect smoothed live model states.")

processor_confidence_threshold = float(
    st.session_state.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
)
processor_closed_frame_threshold = int(
    st.session_state.get("closed_frame_threshold", DEFAULT_CLOSED_FRAMES)
)
processor_ema_alpha = float(st.session_state.get("ema_alpha", DEFAULT_EMA_ALPHA))
processor_process_every_n_frames = int(
    st.session_state.get("process_every_n_frames", DEFAULT_PROCESS_EVERY_N)
)
processor_detect_every_n_frames = int(
    st.session_state.get("detect_every_n_frames", DEFAULT_DETECT_EVERY_N)
)
processor_stream_width = int(st.session_state.get("stream_width", DEFAULT_STREAM_WIDTH))
processor_stream_height = int(st.session_state.get("stream_height", DEFAULT_STREAM_HEIGHT))
processor_stream_fps = int(st.session_state.get("stream_fps", DEFAULT_STREAM_FPS))
ui_refresh_seconds = int(st.session_state.get("ui_refresh_seconds", DEFAULT_UI_REFRESH_SECONDS))


def video_processor_factory():
    return VideoProcessor(
        confidence_threshold=processor_confidence_threshold,
        closed_frame_threshold=processor_closed_frame_threshold,
        ema_alpha=processor_ema_alpha,
        process_every_n_frames=processor_process_every_n_frames,
        detect_every_n_frames=processor_detect_every_n_frames,
    )

webrtc_ctx = webrtc_streamer(
    key="drowsiness-webcam",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=video_processor_factory,
    media_stream_constraints={
        "video": {
            "width": {"ideal": processor_stream_width},
            "height": {"ideal": processor_stream_height},
            "frameRate": {"ideal": processor_stream_fps, "max": processor_stream_fps},
        },
        "audio": False,
    },
    async_processing=True,
)

if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.update_runtime_config(
        confidence_threshold=st.session_state.confidence_threshold,
        closed_frame_threshold=st.session_state.closed_frame_threshold,
        ema_alpha=st.session_state.ema_alpha,
        process_every_n_frames=st.session_state.process_every_n_frames,
        detect_every_n_frames=st.session_state.detect_every_n_frames,
    )

is_playing_now = bool(getattr(webrtc_ctx.state, "playing", False))
if webrtc_ctx.video_processor and st.session_state.was_playing and not is_playing_now:
    webrtc_ctx.video_processor.reset_state()
st.session_state.was_playing = is_playing_now


def render_live_panel():
    metrics = get_display_metrics(webrtc_ctx)
    render_live_cards(metrics)
    st.caption(
        "Drowsy activates only when eyes_closed is sustained for N frames. "
        "No-face frames reset streak counters to avoid stale alerts."
    )


if hasattr(st, "fragment"):
    @st.fragment(run_every=ui_refresh_seconds)
    def live_cards_fragment():
        render_live_panel()


    live_cards_fragment()
else:
    render_live_panel()

