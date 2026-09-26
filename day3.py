from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = str(BASE_DIR / "traffic4.mp4")
OUTPUT_PATH = str(BASE_DIR / "output_traffic5_accident_alert.mp4")

MODEL_PATH = "yolov8n.pt"
TRACKER_CONFIG = "bytetrack.yaml"

# Preview and output settings
SHOW_PREVIEW = True
FULLSCREEN_PREVIEW = True
PREVIEW_WAIT_MS = 30

# Output plays at half speed. All frames are still processed.
SLOW_MOTION_FACTOR = 0.5

# Conservative accident-event heuristics.
# These thresholds are image-based and must be tuned to your camera/video.
CONTACT_IOU_THRESHOLD = 0.10
CONTACT_NORMALIZED_DISTANCE = 0.85
SUDDEN_DECELERATION_RATIO = 0.45
MIN_SPEED_BEFORE_DECELERATION = 35.0  # pixels/second
ACCIDENT_EVIDENCE_WINDOW = 8
ACCIDENT_EVIDENCE_REQUIRED = 5
ALERT_HOLD_SECONDS = 4.0

# The code only evaluates likely road vehicles from COCO classes:
# car=2, motorcycle=3, bus=5, truck=7.
VEHICLE_CLASS_IDS = {2, 3, 5, 7}


# ============================================================
# HELPERS
# ============================================================
def center_of(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def velocity_from_history(history, fps, frame_gap=3):
    """Image-plane velocity (pixels/sec), using original video timing."""
    if len(history) <= frame_gap:
        return None

    p_now = np.array(history[-1][:2], dtype=float)
    p_old = np.array(history[-1 - frame_gap][:2], dtype=float)
    dt = frame_gap / fps

    if dt <= 0:
        return None

    return (p_now - p_old) / dt


def speed_from_history(history, fps, frame_gap=3):
    velocity = velocity_from_history(history, fps, frame_gap)
    if velocity is None:
        return None
    return float(np.linalg.norm(velocity))


def is_sudden_deceleration(history, fps):
    """
    Look for a strong, sustained drop in image-plane speed.
    This is only supporting evidence: perspective and occlusion can affect it.
    """
    if len(history) < 8:
        return False

    earlier = list(history)[:-3]
    recent = list(history)[-4:]

    old_speed = speed_from_history(earlier, fps, frame_gap=3)
    new_speed = speed_from_history(recent, fps, frame_gap=2)

    if old_speed is None or new_speed is None:
        return False

    return (
        old_speed >= MIN_SPEED_BEFORE_DECELERATION
        and new_speed <= old_speed * SUDDEN_DECELERATION_RATIO
    )


def vehicle_pair_has_contact(box_a, box_b):
    """
    Contact candidate requires meaningful box overlap or very close boxes.
    A candidate is not sufficient by itself to declare an accident.
    """
    overlap = box_iou(box_a, box_b)
    center_distance = float(
        np.linalg.norm(center_of(box_a) - center_of(box_b))
    )

    diagonal_a = float(np.hypot(
        max(0, box_a[2] - box_a[0]),
        max(0, box_a[3] - box_a[1]),
    ))
    diagonal_b = float(np.hypot(
        max(0, box_b[2] - box_b[0]),
        max(0, box_b[3] - box_b[1]),
    ))
    average_size = max((diagonal_a + diagonal_b) / 2.0, 1.0)
    normalized_distance = center_distance / average_size

    contact_candidate = (
        overlap >= CONTACT_IOU_THRESHOLD
        or normalized_distance <= CONTACT_NORMALIZED_DISTANCE
    )

    return contact_candidate, overlap, normalized_distance


def draw_label(frame, text, x, y, color, scale=0.8, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x = max(0, min(int(x), frame.shape[1] - tw - 12))
    y = max(th + 12, min(int(y), frame.shape[0] - baseline - 4))

    cv2.rectangle(
        frame, (x, y - th - 10), (x + tw + 12, y + baseline + 2),
        color, -1
    )
    cv2.putText(
        frame, text, (x + 6, y - 4), font, scale,
        (255, 255, 255), thickness, cv2.LINE_AA
    )


# ============================================================
# INPUT / OUTPUT
# ============================================================
if not Path(VIDEO_PATH).is_file():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}\n"
        "Place traffic5.mp4 in the same folder as this script."
    )

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
source_fps = float(cap.get(cv2.CAP_PROP_FPS))
if not np.isfinite(source_fps) or source_fps <= 0:
    source_fps = 30.0

output_fps = max(source_fps * SLOW_MOTION_FACTOR, 1.0)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, output_fps, (width, height))

if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Could not create output video: {OUTPUT_PATH}")

if SHOW_PREVIEW:
    cv2.namedWindow("CCTV Accident Detection", cv2.WINDOW_NORMAL)
    if FULLSCREEN_PREVIEW:
        cv2.setWindowProperty(
            "CCTV Accident Detection",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )

# Track histories contain [center_x, center_y, box_width, box_height].
track_history = defaultdict(lambda: deque(maxlen=30))
# A pair gets an accident candidate only when contact and sudden deceleration
# evidence occur together. Require repeated evidence across multiple frames.
pair_evidence_history = defaultdict(
    lambda: deque(maxlen=ACCIDENT_EVIDENCE_WINDOW)
)

# Track IDs involved in the most recently confirmed event.
active_accident_ids = set()
accident_alert_until = -1.0
event_already_reported = False
frame_index = 0

print(f"Input video: {VIDEO_PATH}")
print(f"Output video: {OUTPUT_PATH}")
print("Processing every frame. Press Q in the preview window to stop.")


# ============================================================
# MAIN LOOP
# ============================================================
try:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        video_time = frame_index / source_fps

        results = model.track(
            frame,
            tracker=TRACKER_CONFIG,
            persist=True,
            verbose=False,
        )
        result = results[0]
        annotated = result.plot(labels=False, conf=False)

        active_boxes = {}
        active_classes = {}

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, class_id in zip(boxes, ids, classes):
                if class_id not in VEHICLE_CLASS_IDS:
                    continue

                x1, y1, x2, y2 = box.astype(int)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                track_history[track_id].append(
                    [cx, cy, x2 - x1, y2 - y1]
                )
                active_boxes[track_id] = np.array(
                    [x1, y1, x2, y2], dtype=int
                )
                active_classes[track_id] = class_id

                draw_label(
                    annotated, f"ID {track_id}", x1, max(25, y1),
                    (35, 125, 35), scale=0.55
                )

        confirmed_event_pairs = []

        # Require contact evidence AND sudden deceleration evidence.
        for id_a, id_b in combinations(active_boxes.keys(), 2):
            pair = tuple(sorted((id_a, id_b)))
            box_a, box_b = active_boxes[id_a], active_boxes[id_b]

            contact, overlap, normalized_distance = vehicle_pair_has_contact(
                box_a, box_b
            )

            decel_a = is_sudden_deceleration(track_history[id_a], source_fps)
            decel_b = is_sudden_deceleration(track_history[id_b], source_fps)

            # Contact without a sudden movement change is not called an accident.
            evidence_this_frame = bool(contact and (decel_a or decel_b))
            pair_evidence_history[pair].append(evidence_this_frame)

            enough_evidence = (
                sum(pair_evidence_history[pair])
                >= ACCIDENT_EVIDENCE_REQUIRED
            )

            if enough_evidence:
                confirmed_event_pairs.append((id_a, id_b, overlap))
                active_accident_ids.update((id_a, id_b))

        if confirmed_event_pairs:
            accident_alert_until = video_time + ALERT_HOLD_SECONDS
            if not event_already_reported:
                involved = sorted(active_accident_ids)
                print(
                    f"[{video_time:.2f}s] ACCIDENT DETECTED - "
                    f"vehicle track IDs: {involved}"
                )
                event_already_reported = True
        elif video_time > accident_alert_until:
            active_accident_ids.clear()
            event_already_reported = False

        # Highlight only the vehicles involved in the confirmed event.
        for track_id in active_accident_ids:
            if track_id in active_boxes:
                x1, y1, x2, y2 = active_boxes[track_id]
                cv2.rectangle(
                    annotated, (x1, y1), (x2, y2), (0, 0, 255), 3
                )

        # Keep the exact requested alert text visible for CCTV viewers.
        if video_time <= accident_alert_until:
            banner_text = "ACCIDENT DETECTED"
            banner_color = (0, 0, 220)
            draw_label(
                annotated, banner_text, 20, 55,
                banner_color, scale=1.1, thickness=3
            )
            ids_text = "Vehicle IDs: " + ", ".join(
                str(i) for i in sorted(active_accident_ids)
            )
            draw_label(
                annotated, ids_text, 20, 100,
                (0, 0, 180), scale=0.7, thickness=2
            )
        else:
            draw_label(
                annotated, "CCTV MONITORING",
                20, 45, (30, 115, 30), scale=0.75
            )

        writer.write(annotated)

        if SHOW_PREVIEW:
            cv2.imshow("CCTV Accident Detection", annotated)
            if cv2.waitKey(PREVIEW_WAIT_MS) & 0xFF == ord("q"):
                print("Stopped early by user.")
                break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

print("Processing complete.")
print(f"Saved video: {OUTPUT_PATH}")
