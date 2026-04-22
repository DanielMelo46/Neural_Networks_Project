import base64
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Drowsiness Detection", layout="wide")

st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

	:root {
		--bg-top: #edf4f7;
		--bg-bottom: #dbe8ee;
		--panel: rgba(255, 255, 255, 0.88);
		--panel-border: rgba(17, 57, 70, 0.14);
		--text-main: #132831;
		--text-muted: #3d5d67;
		--ok: #1f7a4f;
		--warn: #a66400;
		--alert: #a22222;
		--idle: #586c74;
	}

	.stApp {
		background: radial-gradient(circle at top right, #f7fbfc 0%, var(--bg-top) 40%, var(--bg-bottom) 100%);
		color: var(--text-main);
	}

	h1, h2, h3 {
		font-family: 'Space Grotesk', sans-serif;
		letter-spacing: 0.2px;
	}

	body, p, div, label, span {
		font-family: 'IBM Plex Sans', sans-serif;
	}

	[data-testid="stSidebar"] {
		background: linear-gradient(180deg, #173540 0%, #214958 100%);
		border-right: 1px solid rgba(255, 255, 255, 0.1);
	}

	[data-testid="stSidebar"] * {
		color: #f1f7fa;
	}

	[data-testid="stSidebar"] button, 
	[data-testid="stSidebar"] button * {
		color: #132831 !important;
	}

	.hero {
		border: 1px solid var(--panel-border);
		border-radius: 16px;
		padding: 1rem 1.2rem;
		background: linear-gradient(120deg, rgba(255,255,255,0.94) 0%, rgba(255,255,255,0.80) 100%);
		box-shadow: 0 12px 30px rgba(14, 39, 50, 0.12);
		margin-bottom: 0.85rem;
	}

	.hero-kicker {
		text-transform: uppercase;
		letter-spacing: 1.1px;
		font-size: 0.74rem;
		color: #2f5a68;
	}

	.hero-title {
		margin-top: 0.2rem;
		margin-bottom: 0.2rem;
		font-size: 1.6rem;
		font-weight: 700;
		color: var(--text-main);
	}

	.hero-sub {
		color: var(--text-muted);
		font-size: 0.95rem;
	}

	.status-card {
		border: 1px solid var(--panel-border);
		border-left: 7px solid var(--idle);
		border-radius: 14px;
		padding: 0.8rem 1rem;
		background: var(--panel);
		box-shadow: 0 7px 20px rgba(16, 42, 53, 0.09);
		margin-top: 0.7rem;
		margin-bottom: 0.6rem;
	}

	.status-label {
		font-size: 0.74rem;
		text-transform: uppercase;
		letter-spacing: 0.9px;
		color: #4a6470;
	}

	.status-value {
		font-family: 'Space Grotesk', sans-serif;
		font-size: 1.38rem;
		font-weight: 700;
		margin-top: 0.2rem;
	}

	.status-sub {
		color: #36535e;
		margin-top: 0.2rem;
		font-size: 0.92rem;
	}

	.metric-card {
		border: 1px solid var(--panel-border);
		border-radius: 12px;
		background: var(--panel);
		padding: 0.72rem 0.86rem;
		box-shadow: 0 5px 14px rgba(16, 43, 53, 0.08);
		min-height: 96px;
	}

	.metric-label {
		text-transform: uppercase;
		letter-spacing: 1px;
		color: #4b6772;
		font-size: 0.72rem;
	}

	.metric-value {
		margin-top: 0.3rem;
		margin-bottom: 0.18rem;
		font-size: 1.05rem;
		font-family: 'Space Grotesk', sans-serif;
		font-weight: 700;
		color: #13313b;
	}

	.metric-sub {
		color: #415f69;
		font-size: 0.86rem;
	}
	</style>
	""",
	unsafe_allow_html=True,
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = APP_DIR / "best_mobilenetv2_drowsiness.keras"
FACE_ALARM_PATH = PROJECT_DIR / "sounds" / "alarm.wav"
DROWSY_ALARM_PATH = PROJECT_DIR / "sounds" / "alarm1.wav"

IMG_SIZE = 160
CLASS_NAMES = {
	0: "yawn",
	1: "no_yawn",
	2: "eyes_closed",
	3: "eyes_open",
}
EYE_CLASS_IDS = [2, 3]
MOUTH_CLASS_IDS = [0, 1]

DEFAULT_DROWSY_TIME_LIMIT = 1.0
DEFAULT_FACE_MISSING_LIMIT = 0.5
DEFAULT_CONFIDENCE_THRESHOLD = 0.55


def init_session_state() -> None:
	defaults = {
		"desired_playing": False,
		"was_playing": False,
		"drowsy_time_limit": DEFAULT_DROWSY_TIME_LIMIT,
		"face_missing_limit": DEFAULT_FACE_MISSING_LIMIT,
		"confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
	}
	for key, value in defaults.items():
		if key not in st.session_state:
			st.session_state[key] = value


@st.cache_resource
def load_model() -> tf.keras.Model:
	return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_audio_base64(audio_path: str) -> str:
	path = Path(audio_path)
	if not path.exists():
		return ""
	return base64.b64encode(path.read_bytes()).decode("utf-8")


def preprocess_roi_for_model(roi_bgr: np.ndarray, img_size: int = IMG_SIZE) -> Optional[np.ndarray]:
	if roi_bgr is None or roi_bgr.size == 0:
		return None
	resized = cv2.resize(roi_bgr, (img_size, img_size))
	rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
	x = preprocess_input(rgb)
	return np.expand_dims(x, axis=0)


def mouth_box_from_face(face_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
	x, y, w, h = face_box
	mx = x + int(0.18 * w)
	my = y + int(0.58 * h)
	mw = int(0.64 * w)
	mh = int(0.34 * h)
	return (mx, my, mw, mh)


def clip_box(
	x: int,
	y: int,
	w: int,
	h: int,
	frame_w: int,
	frame_h: int,
) -> Optional[tuple[int, int, int, int]]:
	x = max(0, x)
	y = max(0, y)
	w = min(w, frame_w - x)
	h = min(h, frame_h - y)
	if w <= 1 or h <= 1:
		return None
	return (x, y, w, h)


def format_alert_state(metrics: Dict[str, object]) -> tuple[str, str, str]:
	if metrics.get("drowsy_alert_active", False):
		return ("Drowsy Alert", "#a22222", metrics.get("summary", ""))
	if metrics.get("face_alert_active", False):
		return ("Face Alert", "#a66400", metrics.get("summary", ""))
	if metrics.get("face_seen", False):
		return ("Monitoring", "#1f7a4f", metrics.get("summary", ""))
	return ("Idle", "#586c74", metrics.get("summary", ""))


def render_status(metrics: Dict[str, object]) -> None:
	status_text, color, subtext = format_alert_state(metrics)
	st.markdown(
		f"""
		<div class="status-card" style="border-left-color:{color};">
			<div class="status-label">System State</div>
			<div class="status-value" style="color:{color};">{status_text}</div>
			<div class="status-sub">{subtext}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)

	cols = st.columns(4, gap="small")
	with cols[0]:
		st.markdown(
			f"""
			<div class="metric-card">
				<div class="metric-label">Eyes</div>
				<div class="metric-value">{metrics.get("eye_status", "not_detected")}</div>
				<div class="metric-sub">conf: {metrics.get("eye_conf", 0.0):.2f}</div>
			</div>
			""",
			unsafe_allow_html=True,
		)
	with cols[1]:
		st.markdown(
			f"""
			<div class="metric-card">
				<div class="metric-label">Mouth</div>
				<div class="metric-value">{metrics.get("mouth_status", "not_detected")}</div>
				<div class="metric-sub">conf: {metrics.get("mouth_conf", 0.0):.2f}</div>
			</div>
			""",
			unsafe_allow_html=True,
		)
	with cols[2]:
		st.markdown(
			f"""
			<div class="metric-card">
				<div class="metric-label">Drowsy Timer</div>
				<div class="metric-value">{metrics.get("elapsed_drowsy", 0.0):.1f}s</div>
				<div class="metric-sub">limit: {metrics.get("drowsy_time_limit", 0.0):.1f}s</div>
			</div>
			""",
			unsafe_allow_html=True,
		)
	with cols[3]:
		st.markdown(
			f"""
			<div class="metric-card">
				<div class="metric-label">Face Missing</div>
				<div class="metric-value">{metrics.get("elapsed_face_missing", 0.0):.1f}s</div>
				<div class="metric-sub">limit: {metrics.get("face_missing_limit", 0.0):.1f}s</div>
			</div>
			""",
			unsafe_allow_html=True,
		)


def render_alarm_audio(metrics: Dict[str, object], face_alarm_b64: str, drowsy_alarm_b64: str) -> None:
	source_b64 = ""
	if metrics.get("drowsy_alert_active", False):
		source_b64 = drowsy_alarm_b64
	elif metrics.get("face_alert_active", False):
		source_b64 = face_alarm_b64

	if source_b64:
		st.markdown(
			(
				"<audio autoplay loop style='display:none;'>"
				f"<source src='data:audio/wav;base64,{source_b64}' type='audio/wav'>"
				"</audio>"
			),
			unsafe_allow_html=True,
		)


def idle_metrics(message: str) -> Dict[str, object]:
	return {
		"eye_status": "not_detected",
		"eye_conf": 0.0,
		"mouth_status": "not_detected",
		"mouth_conf": 0.0,
		"face_seen": False,
		"face_alert_active": False,
		"drowsy_alert_active": False,
		"elapsed_drowsy": 0.0,
		"elapsed_face_missing": 0.0,
		"drowsy_time_limit": float(st.session_state.get("drowsy_time_limit", DEFAULT_DROWSY_TIME_LIMIT)),
		"face_missing_limit": float(st.session_state.get("face_missing_limit", DEFAULT_FACE_MISSING_LIMIT)),
		"summary": message,
	}


class DrowsinessVideoProcessor(VideoProcessorBase):
	def __init__(self, model: tf.keras.Model, confidence_threshold: float, drowsy_time_limit: float, face_missing_limit: float):
		self.model = model
		self.confidence_threshold = float(confidence_threshold)
		self.drowsy_time_limit = float(drowsy_time_limit)
		self.face_missing_limit = float(face_missing_limit)

		self.lock = threading.Lock()
		self.latest_metrics = {
			"eye_status": "not_detected",
			"eye_conf": 0.0,
			"mouth_status": "not_detected",
			"mouth_conf": 0.0,
			"face_seen": False,
			"face_alert_active": False,
			"drowsy_alert_active": False,
			"elapsed_drowsy": 0.0,
			"elapsed_face_missing": 0.0,
			"drowsy_time_limit": self.drowsy_time_limit,
			"face_missing_limit": self.face_missing_limit,
			"summary": "Allow camera access, then press Start.",
		}

		self.drowsy_start_time = None
		self.face_missing_start_time = None

		self.mp_face_detection = mp.solutions.face_detection
		self.face_detector = self.mp_face_detection.FaceDetection(
			model_selection=0,
			min_detection_confidence=0.5,
		)

	def update_settings(self, confidence_threshold: float, drowsy_time_limit: float, face_missing_limit: float) -> None:
		with self.lock:
			self.confidence_threshold = float(confidence_threshold)
			self.drowsy_time_limit = float(drowsy_time_limit)
			self.face_missing_limit = float(face_missing_limit)

	def reset_state(self, message: str = "Stopped. Press Start to monitor again.") -> None:
		with self.lock:
			self.drowsy_start_time = None
			self.face_missing_start_time = None
			self.latest_metrics = {
				"eye_status": "not_detected",
				"eye_conf": 0.0,
				"mouth_status": "not_detected",
				"mouth_conf": 0.0,
				"face_seen": False,
				"face_alert_active": False,
				"drowsy_alert_active": False,
				"elapsed_drowsy": 0.0,
				"elapsed_face_missing": 0.0,
				"drowsy_time_limit": self.drowsy_time_limit,
				"face_missing_limit": self.face_missing_limit,
				"summary": message,
			}

	def get_latest_metrics(self) -> Dict[str, object]:
		with self.lock:
			return dict(self.latest_metrics)

	def _get_status_from_probs(self, probs: np.ndarray, allowed_ids: list[int]) -> tuple[str, float]:
		subset = np.array([probs[i] for i in allowed_ids], dtype=np.float32)
		subset = subset / (subset.sum() + 1e-8)

		best_local_idx = int(np.argmax(subset))
		class_id = allowed_ids[best_local_idx]
		confidence = float(subset[best_local_idx])

		if confidence < self.confidence_threshold:
			return "uncertain", confidence
		return CLASS_NAMES[class_id], confidence

	def detect_and_annotate_states(self, image_bgr: np.ndarray) -> tuple[np.ndarray, str, float, str, float, bool]:
		output = image_bgr.copy()
		frame_h, frame_w = image_bgr.shape[:2]

		rgb_in = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		results = self.face_detector.process(rgb_in)

		if not results or not results.detections:
			return output, "not_detected", 0.0, "not_detected", 0.0, False

		detection = max(
			results.detections,
			key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height,
		)
		bbox = detection.location_data.relative_bounding_box
		fx = int(bbox.xmin * frame_w)
		fy = int(bbox.ymin * frame_h)
		fw = int(bbox.width * frame_w)
		fh = int(bbox.height * frame_h)

		clipped_face = clip_box(fx, fy, fw, fh, frame_w, frame_h)
		if clipped_face is None:
			return output, "not_detected", 0.0, "not_detected", 0.0, False
		fx, fy, fw, fh = clipped_face
		face_box = (fx, fy, fw, fh)

		eye_zone_h = max(2, int(fh * 0.45))
		eye_box = clip_box(fx, fy, fw, eye_zone_h, frame_w, frame_h)
		eye_roi = None if eye_box is None else image_bgr[eye_box[1] : eye_box[1] + eye_box[3], eye_box[0] : eye_box[0] + eye_box[2]]

		mx, my, mw, mh = mouth_box_from_face(face_box)
		mouth_box = clip_box(mx, my, mw, mh, frame_w, frame_h)
		mouth_roi = None if mouth_box is None else image_bgr[mouth_box[1] : mouth_box[1] + mouth_box[3], mouth_box[0] : mouth_box[0] + mouth_box[2]]

		eye_status, eye_conf = "not_detected", 0.0
		mouth_status, mouth_conf = "not_detected", 0.0

		x_eye = preprocess_roi_for_model(eye_roi)
		x_mouth = preprocess_roi_for_model(mouth_roi)

		batch = []
		if x_eye is not None:
			batch.append(x_eye[0])
		if x_mouth is not None:
			batch.append(x_mouth[0])

		if batch:
			batch_tensor = np.array(batch, dtype=np.float32)
			# Fast batched inference avoiding tf.data overhead
			probs = self.model(batch_tensor, training=False).numpy()

			idx = 0
			if x_eye is not None:
				eye_status, eye_conf = self._get_status_from_probs(probs[idx], EYE_CLASS_IDS)
				idx += 1
			if x_mouth is not None:
				mouth_status, mouth_conf = self._get_status_from_probs(probs[idx], MOUTH_CLASS_IDS)

		is_drowsy = eye_status == "eyes_closed" or mouth_status == "yawn"
		color = (0, 0, 255) if is_drowsy else (0, 255, 0)

		cv2.rectangle(output, (fx, fy), (fx + fw, fy + fh), color, 2)
		cv2.putText(
			output,
			f"E: {eye_status} | M: {mouth_status}",
			(fx, max(20, fy - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.55,
			color,
			2,
			cv2.LINE_AA,
		)

		return output, eye_status, eye_conf, mouth_status, mouth_conf, True

	def apply_alarm_logic(
		self,
		annotated: np.ndarray,
		eye_status: str,
		mouth_status: str,
		face_seen: bool,
	) -> tuple[np.ndarray, Dict[str, object]]:
		now = time.time()
		frame_h, frame_w = annotated.shape[:2]

		face_alert_active = False
		drowsy_alert_active = False
		elapsed_face_missing = 0.0
		elapsed_drowsy = 0.0

		if not face_seen:
			if self.face_missing_start_time is None:
				self.face_missing_start_time = now
			elapsed_face_missing = now - self.face_missing_start_time

			if elapsed_face_missing >= self.face_missing_limit:
				face_alert_active = True
				cv2.putText(
					annotated,
					"FACE ALERT: NOT DETECTED",
					(frame_w // 8, frame_h // 2),
					cv2.FONT_HERSHEY_SIMPLEX,
					1.0,
					(0, 165, 255),
					3,
					cv2.LINE_AA,
				)

			self.drowsy_start_time = None
		else:
			self.face_missing_start_time = None

		is_drowsy_state = eye_status == "eyes_closed" or mouth_status == "yawn"
		if face_seen and is_drowsy_state:
			if self.drowsy_start_time is None:
				self.drowsy_start_time = now

			elapsed_drowsy = now - self.drowsy_start_time
			warning_msg = "EYES CLOSED" if eye_status == "eyes_closed" else "YAWNING"
			cv2.putText(
				annotated,
				f"{warning_msg} {elapsed_drowsy:.1f}s",
				(10, 40),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.85,
				(0, 0, 255),
				2,
				cv2.LINE_AA,
			)

			if elapsed_drowsy >= self.drowsy_time_limit:
				drowsy_alert_active = True
				cv2.putText(
					annotated,
					"WAKE UP",
					(frame_w // 3, frame_h // 2 + 80),
					cv2.FONT_HERSHEY_SIMPLEX,
					1.8,
					(0, 0, 255),
					4,
					cv2.LINE_AA,
				)
		else:
			self.drowsy_start_time = None

		if drowsy_alert_active:
			summary = "Drowsy alert active. Eyes closed or yawning persisted beyond limit."
		elif face_alert_active:
			summary = "Face alert active. Face missing beyond limit."
		elif face_seen:
			summary = "Monitoring active. Face detected and within thresholds."
		else:
			summary = "Waiting for face."

		metrics = {
			"eye_status": eye_status,
			"eye_conf": 0.0,
			"mouth_status": mouth_status,
			"mouth_conf": 0.0,
			"face_seen": face_seen,
			"face_alert_active": face_alert_active,
			"drowsy_alert_active": drowsy_alert_active,
			"elapsed_drowsy": float(elapsed_drowsy),
			"elapsed_face_missing": float(elapsed_face_missing),
			"drowsy_time_limit": float(self.drowsy_time_limit),
			"face_missing_limit": float(self.face_missing_limit),
			"summary": summary,
		}
		return annotated, metrics

	def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
		image_bgr = frame.to_ndarray(format="bgr24")
		annotated, eye_status, eye_conf, mouth_status, mouth_conf, face_seen = self.detect_and_annotate_states(image_bgr)
		annotated, metrics = self.apply_alarm_logic(annotated, eye_status, mouth_status, face_seen)
		metrics["eye_conf"] = eye_conf
		metrics["mouth_conf"] = mouth_conf

		with self.lock:
			self.latest_metrics = metrics

		return av.VideoFrame.from_ndarray(annotated, format="bgr24")


init_session_state()

st.markdown(
	"""
	<section class="hero">
		<div class="hero-kicker">Driver Monitoring</div>
		<div class="hero-title">Drowsiness Detection</div>
		<div class="hero-sub">Webcam Test For Neural Networks Project COMP-3704	.</div>
	</section>
	""",
	unsafe_allow_html=True,
)

try:
	model_for_streamlit = load_model()
except Exception as exc:
	st.error(f"Model could not be loaded from {MODEL_PATH}: {exc}")
	st.stop()

face_alarm_b64 = load_audio_base64(str(FACE_ALARM_PATH))
drowsy_alarm_b64 = load_audio_base64(str(DROWSY_ALARM_PATH))

with st.sidebar:
	st.header("Controls")
	st.caption("Use Start and Stop to control webcam monitoring.")

	start_col, stop_col = st.columns(2)
	with start_col:
		if st.button("Start", use_container_width=True):
			st.session_state.desired_playing = True
	with stop_col:
		if st.button("Stop", use_container_width=True):
			st.session_state.desired_playing = False

	st.markdown("### Alarm Thresholds")
	st.slider(
		"Drowsy Time Limit (seconds)",
		min_value=0.5,
		max_value=5.0,
		step=0.1,
		key="drowsy_time_limit",
	)
	st.slider(
		"Face Missing Limit (seconds)",
		min_value=0.2,
		max_value=3.0,
		step=0.1,
		key="face_missing_limit",
	)
	st.slider(
		"Confidence Threshold",
		min_value=0.40,
		max_value=0.95,
		step=0.01,
		key="confidence_threshold",
	)

	if bool(st.session_state.get("desired_playing", False)):
		st.success("Camera requested: ON")
	else:
		st.info("Camera requested: OFF")

	if not face_alarm_b64 or not drowsy_alarm_b64:
		st.warning("One or more alarm sound files were not found. Visual alerts will still work.")


desired_playing = bool(st.session_state.get("desired_playing", False))
confidence_threshold_value = float(st.session_state.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
drowsy_time_limit_value = float(st.session_state.get("drowsy_time_limit", DEFAULT_DROWSY_TIME_LIMIT))
face_missing_limit_value = float(st.session_state.get("face_missing_limit", DEFAULT_FACE_MISSING_LIMIT))


def video_processor_factory() -> DrowsinessVideoProcessor:
	return DrowsinessVideoProcessor(
		model=model_for_streamlit,
		confidence_threshold=confidence_threshold_value,
		drowsy_time_limit=drowsy_time_limit_value,
		face_missing_limit=face_missing_limit_value,
	)


webrtc_ctx = webrtc_streamer(
	key="drowsiness-v1-stream",
	mode=WebRtcMode.SENDRECV,
	desired_playing_state=desired_playing,
	media_stream_constraints={
		"video": {
			"width": {"ideal": 640},
			"height": {"ideal": 480},
			"frameRate": {"ideal": 20, "max": 24},
		},
		"audio": False,
	},
	video_processor_factory=video_processor_factory,
	async_processing=True,
)

if webrtc_ctx.video_processor:
	webrtc_ctx.video_processor.update_settings(
		confidence_threshold=confidence_threshold_value,
		drowsy_time_limit=drowsy_time_limit_value,
		face_missing_limit=face_missing_limit_value,
	)

is_playing_now = bool(getattr(webrtc_ctx.state, "playing", False))
if webrtc_ctx.video_processor and bool(st.session_state.get("was_playing", False)) and not is_playing_now:
	webrtc_ctx.video_processor.reset_state()
st.session_state.was_playing = is_playing_now


def get_metrics_for_ui() -> Dict[str, object]:
	if bool(getattr(webrtc_ctx.state, "playing", False)) and webrtc_ctx.video_processor:
		return webrtc_ctx.video_processor.get_latest_metrics()
	if webrtc_ctx.video_processor:
		return idle_metrics("Webcam stopped. Press Start to begin monitoring.")
	return idle_metrics("Allow camera access and press Start.")


def render_live_ui() -> None:
	metrics = get_metrics_for_ui()
	render_status(metrics)
	render_alarm_audio(metrics, face_alarm_b64, drowsy_alarm_b64)


if hasattr(st, "fragment"):
	@st.fragment(run_every=1)
	def live_fragment() -> None:
		render_live_ui()


	live_fragment()
else:
	render_live_ui()