# Real-Time Driver Drowsiness Detection

A deep learning project that detects signs of driver drowsiness using a webcam and triggers audio alerts when the driver appears sleepy or their face is missing from the frame.

## How It Works

- **Face detection:** MediaPipe locates the driver's face.
- **Feature extraction:** OpenCV crops the eye and mouth regions.
- **Classification:** A fine-tuned MobileNetV2 model classifies four states: eyes open, eyes closed, yawning, and no yawning.
- **Alert system:** Time-based thresholds reduce false alarms from brief blinks or mouth movements. Separate audio alerts indicate drowsiness and a missing face.
- **Web interface:** A Streamlit dashboard displays the live video, predictions, confidence scores, and alert status.

## Tech Stack

Python, TensorFlow/Keras, MobileNetV2, OpenCV, MediaPipe, Streamlit, streamlit-webrtc, Pygame.

## Results

- Approximately **97% test accuracy** reported after fine-tuning.
- Approximately **98% validation accuracy** during fine-tuning.
- Strong classification results for eye states and yawning.
- Improved real-time detection through region-of-interest cropping, confidence thresholds, and time-based filtering.

## Future Improvements

- Improve detection under different lighting conditions.
- Test heavier architectures such as ResNet-50.
- Further reduce false alarms before considering additional vehicle safety features.

*This project is an academic prototype and is not a replacement for certified driver safety systems.*
