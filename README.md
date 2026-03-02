# Virtual Try-On Webcam Filter

A real-time Python app that uses your webcam and MediaPipe Pose to overlay a virtual cloth on the upper body.

## Features
- Live webcam processing
- Pose landmark tracking with MediaPipe
- Perspective cloth overlay with adjustable opacity
- Optional mirrored preview (`--flip`)
- Fallback random cloth generation when image is missing

## Requirements
- Python 3.10+
- `opencv-python`
- `numpy`
- `mediapipe`

Install dependencies:

```bash
pip install opencv-python numpy mediapipe
```

## Run

```bash
python webcam_ascii.py --cloth cloth.png --opacity 1.0 --flip
```

## Controls
- `q` or `Esc`: quit
- `f`: toggle mirror
- `+` / `-`: increase/decrease opacity
- `r`: generate random cloth

## Notes
- If the cloth image path is invalid, the app automatically switches to a generated random cloth.
- Best results are achieved with good lighting and when shoulders/hips are visible.
