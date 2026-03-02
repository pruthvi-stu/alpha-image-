import argparse
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise SystemExit(
        "mediapipe is required. Install with: pip install mediapipe"
    ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live virtual clothing try-on filter")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    parser.add_argument(
        "--cloth",
        type=str,
        default="cloth.png",
        help="Path to clothing PNG/JPG (PNG with transparency recommended). Uses random cloth if missing.",
    )
    parser.add_argument("--opacity", type=float, default=1.0, help="Overlay opacity (0.0-1.0)")
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.45,
        help="Minimum landmark visibility to apply filter",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe pose model complexity",
    )
    parser.add_argument("--flip", action="store_true", help="Mirror camera view")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed for random cloth generation (only used when cloth image is missing)",
    )
    return parser


def load_cloth(cloth_path: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(cloth_path)
    cloth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if cloth is None:
        raise FileNotFoundError(f"Could not load cloth image: {path}")

    if cloth.ndim == 2:
        cloth_bgr = cv2.cvtColor(cloth, cv2.COLOR_GRAY2BGR)
        alpha = np.full(cloth.shape, 255, dtype=np.uint8)
    elif cloth.shape[2] == 4:
        cloth_bgr = cloth[:, :, :3]
        alpha = cloth[:, :, 3]
    else:
        cloth_bgr = cloth[:, :, :3]
        alpha = np.full(cloth_bgr.shape[:2], 255, dtype=np.uint8)

    return cloth_bgr, alpha


def generate_random_cloth(
    width: int = 520,
    height: int = 620,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    base_color = rng.integers(40, 230, size=3, dtype=np.uint8)
    cloth = np.full((height, width, 3), base_color, dtype=np.uint8)

    pattern_type = int(rng.integers(0, 3))
    if pattern_type == 0:
        for x in range(0, width, int(rng.integers(18, 48))):
            stripe_color = rng.integers(0, 255, size=3, dtype=np.uint8)
            stripe_w = int(rng.integers(8, 20))
            cv2.rectangle(cloth, (x, 0), (min(width - 1, x + stripe_w), height - 1), tuple(int(v) for v in stripe_color), -1)
    elif pattern_type == 1:
        for y in range(0, height, int(rng.integers(20, 50))):
            line_color = rng.integers(0, 255, size=3, dtype=np.uint8)
            line_thick = int(rng.integers(3, 10))
            cv2.line(cloth, (0, y), (width - 1, y), tuple(int(v) for v in line_color), line_thick)
    else:
        for _ in range(int(rng.integers(35, 90))):
            center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            radius = int(rng.integers(6, 28))
            dot_color = rng.integers(0, 255, size=3, dtype=np.uint8)
            cv2.circle(cloth, center, radius, tuple(int(v) for v in dot_color), -1)

    alpha = np.zeros((height, width), dtype=np.uint8)
    torso_pts = np.array(
        [
            (int(width * 0.20), int(height * 0.10)),
            (int(width * 0.80), int(height * 0.10)),
            (int(width * 0.94), int(height * 0.90)),
            (int(width * 0.06), int(height * 0.90)),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(alpha, torso_pts, 255)
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
    return cloth, alpha


def to_px(landmark, frame_w: int, frame_h: int) -> np.ndarray:
    return np.array([landmark.x * frame_w, landmark.y * frame_h], dtype=np.float32)


def get_torso_quad(landmarks, frame_w: int, frame_h: int, min_visibility: float) -> np.ndarray | None:
    pose = mp.solutions.pose.PoseLandmark

    ids = {
        "ls": pose.LEFT_SHOULDER.value,
        "rs": pose.RIGHT_SHOULDER.value,
        "lh": pose.LEFT_HIP.value,
        "rh": pose.RIGHT_HIP.value,
    }

    for idx in ids.values():
        if landmarks[idx].visibility < min_visibility:
            return None

    ls = to_px(landmarks[ids["ls"]], frame_w, frame_h)
    rs = to_px(landmarks[ids["rs"]], frame_w, frame_h)
    lh = to_px(landmarks[ids["lh"]], frame_w, frame_h)
    rh = to_px(landmarks[ids["rh"]], frame_w, frame_h)

    shoulder_vec = rs - ls
    shoulder_w = np.linalg.norm(shoulder_vec)
    if shoulder_w < 8:
        return None

    shoulder_center = (ls + rs) * 0.5
    hip_center = (lh + rh) * 0.5
    torso_h = np.linalg.norm(hip_center - shoulder_center)
    if torso_h < 12:
        return None

    shoulder_dir = shoulder_vec / shoulder_w
    x_pad = 0.18 * shoulder_w
    top_pad = 0.32 * torso_h
    bottom_pad = 0.24 * torso_h

    top_left = ls - shoulder_dir * x_pad + np.array([0.0, -top_pad], dtype=np.float32)
    top_right = rs + shoulder_dir * x_pad + np.array([0.0, -top_pad], dtype=np.float32)
    bottom_right = rh + shoulder_dir * x_pad + np.array([0.0, bottom_pad], dtype=np.float32)
    bottom_left = lh - shoulder_dir * x_pad + np.array([0.0, bottom_pad], dtype=np.float32)

    quad = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    quad[:, 0] = np.clip(quad[:, 0], 0, frame_w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, frame_h - 1)
    return quad


def apply_overlay(
    frame: np.ndarray,
    cloth_bgr: np.ndarray,
    cloth_alpha: np.ndarray,
    quad: np.ndarray,
    opacity: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    src_h, src_w = cloth_bgr.shape[:2]

    src = np.array(
        [[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(src, quad)
    warped_bgr = cv2.warpPerspective(
        cloth_bgr,
        transform,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_alpha = cv2.warpPerspective(
        cloth_alpha,
        transform,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    alpha = (warped_alpha.astype(np.float32) / 255.0) * float(np.clip(opacity, 0.0, 1.0))
    alpha_3 = alpha[:, :, None]

    out = frame.astype(np.float32) * (1.0 - alpha_3) + warped_bgr.astype(np.float32) * alpha_3
    return out.astype(np.uint8)


def open_camera(index: int) -> cv2.VideoCapture | None:
    ordered_indices = [index] + [i for i in range(3) if i != index]
    for idx in ordered_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[Info] Using camera index {idx}")
            return cap
        cap.release()
    return None


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.random_seed)

    try:
        cloth_bgr, cloth_alpha = load_cloth(args.cloth)
        cloth_mode = "image"
    except FileNotFoundError:
        cloth_bgr, cloth_alpha = generate_random_cloth(rng=rng)
        cloth_mode = "random"
        print(f"[Info] Cloth image not found ({args.cloth}). Using random cloth.")
    cap = open_camera(args.camera_index)
    if cap is None:
        raise SystemExit("[Error] Could not open webcam.")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    opacity = float(np.clip(args.opacity, 0.0, 1.0))
    mirror = args.flip
    frame_count = 0
    t0 = time.time()

    print("[Info] Controls: q/ESC quit, f flip, +/- opacity, r random cloth")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            if mirror:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                quad = get_torso_quad(result.pose_landmarks.landmark, w, h, args.min_visibility)
                if quad is not None:
                    frame = apply_overlay(frame, cloth_bgr, cloth_alpha, quad, opacity)
                    for p in quad.astype(np.int32):
                        cv2.circle(frame, tuple(p), 3, (80, 200, 255), -1)

            frame_count += 1
            fps = frame_count / max(time.time() - t0, 1e-3)
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | opacity: {opacity:.2f} | cloth: {cloth_mode}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Virtual Try-On", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("f"):
                mirror = not mirror
            if key in (ord("+"), ord("=")):
                opacity = min(1.0, opacity + 0.05)
            if key in (ord("-"), ord("_")):
                opacity = max(0.0, opacity - 0.05)
            if key == ord("r"):
                cloth_bgr, cloth_alpha = generate_random_cloth(rng=rng)
                cloth_mode = "random"
    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()
        print("[Info] Camera released.")


if __name__ == "__main__":
    main()
