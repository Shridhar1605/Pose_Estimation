import os
import time
import cv2
import numpy as np
import torch
import onnxruntime as ort
import torchvision

# dataset link: https://www.kaggle.com/datasets/fmena14/crowd-counting
from platform_utils import DEVICE, device_name, make_ocsort, make_ort_session  # CUDA -> MPS (macOS) -> CPU
print(f"Using device: {DEVICE} ({device_name()})")

# ---------------------------------------------------------------------------
# PeopleNet (ResNet34 INT8 ONNX) detector — replaces YOLO
# Config sourced from _/nvinfer_config.txt:
#   net-scale-factor = 1/255,  offsets = 0,0,0
#   infer-dims = 3;544;960,    num-detected-classes = 3 (person/bag/face)
#   model-color-format = 0 (BGR)
# ---------------------------------------------------------------------------
class PeopleNetDetector:
    PERSON_CLASS = 0          # class index for 'person' in PeopleNet
    STRIDE       = 16         # DetectNet_v2 default spatial stride
    INPUT_W      = 960
    INPUT_H      = 544
    SCALE        = 1.0 / 255.0

    def __init__(self, model_path="_/resnet34_peoplenet_int8.onnx"):
        # Cross-platform session: CUDA (Linux/Windows) -> CoreML (macOS) -> CPU
        self.session = make_ort_session(model_path, log_prefix="[PeopleNet]")
        self.input_name = self.session.get_inputs()[0].name
        # Identify output tensors: PeopleNet emits coverage map and bbox map
        out_names = [o.name for o in self.session.get_outputs()]
        print(f"PeopleNet loaded | outputs: {out_names} | providers: {self.session.get_providers()}")
        # Determine which output is coverage (3-channel) and which is bbox (12-channel)
        out_shapes = [self.session.get_outputs()[i].shape for i in range(len(out_names))]
        self._cov_idx, self._bbox_idx = self._resolve_output_indices(out_shapes)

    @staticmethod
    def _resolve_output_indices(shapes):
        """Return (cov_idx, bbox_idx) by inspecting the channel dimension."""
        for i, s in enumerate(shapes):
            # shape is [batch, channels, grid_h, grid_w]; channels=3 → cov, =12 → bbox
            if len(s) == 4 and s[1] in (3, 12):
                if s[1] == 3:
                    return i, 1 - i
        return 0, 1  # sensible fallback

    def preprocess(self, bgr_frame):
        """Letterbox resize to [1, 3, 544, 960] while preserving aspect ratio."""
        shape = bgr_frame.shape[:2]  # [height, width]
        
        # Scale ratio (new / old)
        r = min(self.INPUT_W / shape[1], self.INPUT_H / shape[0])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.INPUT_W - new_unpad[0], self.INPUT_H - new_unpad[1]  # wh padding
        
        # Divide padding into 2 sides
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            resized = cv2.resize(bgr_frame, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = bgr_frame
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        blob = padded.astype(np.float32) * self.SCALE          # [H, W, 3]
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]         # [1, 3, H, W]
        return blob, r, dw, dh

    def detect(self, bgr_frame, conf_threshold=0.4, nms_iou=0.45):
        """
        Run PeopleNet inference on a single BGR frame.
        Returns ndarray of shape (N, 5): [x1, y1, x2, y2, score] in original-frame pixels.
        """
        orig_h, orig_w = bgr_frame.shape[:2]
        blob, r, dw, dh = self.preprocess(bgr_frame)

        outputs  = self.session.run(None, {self.input_name: blob})
        cov_map  = outputs[self._cov_idx]   # [1, 3, grid_h, grid_w]
        bbox_map = outputs[self._bbox_idx]  # [1, 12, grid_h, grid_w]

        # --- person class (class 0) coverage and bbox channels ---
        person_scores = cov_map[0, self.PERSON_CLASS]         # [grid_h, grid_w]
        c = self.PERSON_CLASS
        bx1_map = bbox_map[0, c * 4 + 0]                     # [grid_h, grid_w]
        by1_map = bbox_map[0, c * 4 + 1]
        bx2_map = bbox_map[0, c * 4 + 2]
        by2_map = bbox_map[0, c * 4 + 3]

        grid_h, grid_w = person_scores.shape
        gy_idx, gx_idx = np.where(person_scores > conf_threshold)

        if len(gy_idx) == 0:
            return np.empty((0, 5))

        # DetectNet_v2: bbox values are relative to cell center, scaled by 35.0
        NORM = 35.0
        # Channels: 0=Left, 1=Top, 2=Right, 3=Bottom
        L = bx1_map[gy_idx, gx_idx]
        T = by1_map[gy_idx, gx_idx]
        R = bx2_map[gy_idx, gx_idx]
        B = by2_map[gy_idx, gx_idx]
        scores = person_scores[gy_idx, gx_idx]

        # Cell centers
        cx = gx_idx * self.STRIDE + self.STRIDE / 2.0
        cy = gy_idx * self.STRIDE + self.STRIDE / 2.0

        x1s = cx - L * NORM
        y1s = cy - T * NORM
        x2s = cx + R * NORM
        y2s = cy + B * NORM

        # Clip to padded image resolution
        x1s = np.clip(x1s, 0, self.INPUT_W)
        y1s = np.clip(y1s, 0, self.INPUT_H)
        x2s = np.clip(x2s, 0, self.INPUT_W)
        y2s = np.clip(y2s, 0, self.INPUT_H)

        # Scale bboxes back to original frame resolution using letterbox parameters
        x1s = (x1s - dw) / r
        x2s = (x2s - dw) / r
        y1s = (y1s - dh) / r
        y2s = (y2s - dh) / r

        # Ensure valid boxes (x2 > x1, y2 > y1) and apply NMS
        valid = (x2s > x1s) & (y2s > y1s)
        if not np.any(valid):
            return np.empty((0, 5))

        boxes_t  = torch.from_numpy(
            np.stack([x1s[valid], y1s[valid], x2s[valid], y2s[valid]], axis=1).astype(np.float32)
        )
        scores_t = torch.from_numpy(scores[valid].astype(np.float32))
        keep     = torchvision.ops.nms(boxes_t, scores_t, nms_iou)

        kept_boxes  = boxes_t[keep].numpy()
        kept_scores = scores_t[keep].numpy().reshape(-1, 1)
        return np.concatenate([kept_boxes, kept_scores], axis=1)   # (N, 5)


peoplenet = PeopleNetDetector("_/resnet34_peoplenet_int8.onnx")

class RTMPoseWrapper:
    def __init__(self, model_path="rtmpose-s.onnx"):
        self.model_path = model_path
        self.session = None
        if os.path.exists(self.model_path):
            self.session = make_ort_session(self.model_path, log_prefix="[RTMPose]")
            print(f"Loaded RTMPose from {self.model_path} | providers: {self.session.get_providers()}")
        else:
            print(f"Warning: {self.model_path} not found. Using dummy keypoints for testing.")

    def infer(self, roi):
        if self.session is None:
            # Return dummy keypoints if no model (17x3: x, y, conf)
            h, w = roi.shape[:2]
            dummy_kps = np.zeros((17, 3))
            dummy_kps[:, 0] = w / 2  # x
            dummy_kps[:, 1] = h / 2  # y
            dummy_kps[:, 2] = 0.9    # conf
            # Shoulders (index 5, 6)
            dummy_kps[5, 1] = h * 0.2
            dummy_kps[6, 1] = h * 0.2
            # Hips (index 11, 12)
            dummy_kps[11, 1] = h * 0.6
            dummy_kps[12, 1] = h * 0.6
            return dummy_kps, np.ones(17) * 0.9
        # Real inference
        h, w = roi.shape[:2]
        
        input_size = (192, 256) # W, H
        resized = cv2.resize(roi, input_size)
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1) # HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img})
        simcc_x, simcc_y = outputs[0][0], outputs[1][0] # (17, 384), (17, 512)
        
        x_locs = np.argmax(simcc_x, axis=1)
        y_locs = np.argmax(simcc_y, axis=1)
        
        scores_x = np.max(simcc_x, axis=1)
        scores_y = np.max(simcc_y, axis=1)
        scores = (scores_x + scores_y) / 2
        
        keypoints = np.zeros((17, 3))
        keypoints[:, 0] = x_locs / (192 * 2) * w
        keypoints[:, 1] = y_locs / (256 * 2) * h
        keypoints[:, 2] = scores
        
        return keypoints, scores

rtmpose = RTMPoseWrapper()

def classify_action(keypoints, keypoint_history, bbox_height):
    """
    Classifies person action into STANDING, SITTING, FIGHTING, or LYING DOWN
    based on RTMPose 17-keypoint output.

    Keypoint indices (COCO):
      0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
      5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
      9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
      13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """
    if keypoints is None or len(keypoints) < 17:
        return "UNKNOWN", 0.0

    def visible(idx, thresh=0.25):
        return keypoints[idx][2] > thresh

    def pt(idx):
        return keypoints[idx][:2]  # x, y

    # --- Derived geometry ---
    shoulders_vis = visible(5) and visible(6)
    hips_vis      = visible(11) and visible(12)
    knees_vis     = visible(13) and visible(14)
    ankles_vis    = visible(15) and visible(16)

    # Midpoints
    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2 if shoulders_vis else None
    hip_y      = (keypoints[11][1] + keypoints[12][1]) / 2 if hips_vis else None
    knee_y     = (keypoints[13][1] + keypoints[14][1]) / 2 if knees_vis else None
    ankle_y    = (keypoints[15][1] + keypoints[16][1]) / 2 if ankles_vis else None

    shoulder_x = (keypoints[5][0] + keypoints[6][0]) / 2 if shoulders_vis else None
    hip_x      = (keypoints[11][0] + keypoints[12][0]) / 2 if hips_vis else None

    bh = bbox_height if bbox_height > 1 else 1

    # --- LYING DOWN ---
    # Shoulders and hips are nearly at the same vertical level
    if shoulders_vis and hips_vis:
        vertical_delta = abs(shoulder_y - hip_y) / bh
        if vertical_delta < 0.24:
            return "LYING DOWN", min(1.0, 1.0 - vertical_delta / 0.24)

    # --- SITTING ---
    # Hips are much lower than shoulders but knees are bent upward (knee_y < hip_y)
    if shoulders_vis and hips_vis and knees_vis:
        hip_to_shoulder = (hip_y - shoulder_y) / bh      # positive means hips below shoulders
        knee_to_hip     = (hip_y - knee_y) / bh          # positive means knees above hips (seated)
        if hip_to_shoulder > 0.15 and knee_to_hip > -0.05:  # hips above ankles, knees not fully extended down
            # Extra check: if ankles visible, sitting means ankles roughly at knee level or above
            if ankle_y is not None:
                knee_ankle_gap = (ankle_y - knee_y) / bh
                if knee_ankle_gap < 0.25:  # ankles not much below knees -> seated
                    return "SITTING", 0.75
            else:
                return "SITTING", 0.65

    # --- FIGHTING ---
    # Arms raised above shoulders + rapid wrist/elbow movement in recent history
    arms_raised = False
    if visible(7) and visible(8) and shoulders_vis:
        elbow_y = (keypoints[7][1] + keypoints[8][1]) / 1.5
        if elbow_y < shoulder_y:  # elbows above shoulders
            arms_raised = True
    if not arms_raised and (visible(9) or visible(10)):
        wrist_ys = []
        if visible(9): wrist_ys.append(keypoints[9][1])
        if visible(10): wrist_ys.append(keypoints[10][1])
        if shoulder_y is not None and any(w < shoulder_y for w in wrist_ys):
            arms_raised = True

    if arms_raised and len(keypoint_history) >= 5:
        # Check wrist velocity over recent frames for rapid movement
        wrist_positions = []
        for entry in keypoint_history[-8:]:
            kps = entry.get("kps")
            if kps is not None and len(kps) >= 17:
                wx = (kps[9][0] + kps[10][0]) / 2
                wy = (kps[9][1] + kps[10][1]) / 2
                wrist_positions.append((wx, wy))
        if len(wrist_positions) >= 3:
            dists = [((wrist_positions[i][0] - wrist_positions[i-1][0])**2 +
                      (wrist_positions[i][1] - wrist_positions[i-1][1])**2)**0.5
                     for i in range(1, len(wrist_positions))]
            avg_speed = sum(dists) / len(dists)
            if avg_speed > 6.0:
                return "FIGHTING", min(1.0, avg_speed / 25.0)

    # --- STANDING (default upright pose) ---
    if shoulders_vis and hips_vis:
        upright_ratio = (hip_y - shoulder_y) / bh
        if upright_ratio > 0.15:
            return "STANDING", min(1.0, upright_ratio / 0.5)

    return "STANDING", 0.5

def detect_persons_robust(image):
    # PeopleNet (ResNet34 INT8 ONNX via onnxruntime-gpu) inference
    return peoplenet.detect(image, conf_threshold=0.4, nms_iou=0.45)

class OCSortTracker:
    def __init__(self, iou_threshold=0.25, max_lost=60, min_confidence=0.25):
        # boxmot-version-agnostic factory (handles 21.x and 25.x import paths / kwargs)
        self.tracker = make_ocsort(iou_threshold, max_lost, min_confidence)

    def update(self, detections, frame=None):
        if len(detections) == 0:
            dets_np = np.empty((0, 6), dtype=np.float32)
        else:
            dets_np = np.array(detections, dtype=np.float32)
            cls_col = np.zeros((dets_np.shape[0], 1), dtype=np.float32)
            dets_np = np.concatenate([dets_np, cls_col], axis=1)

        if frame is None:
            frame = np.zeros((100, 100, 3), dtype=np.uint8)

        res = self.tracker.update(dets_np, frame)
        
        tracked = []
        for r in res:
            x1, y1, x2, y2, track_id, conf = r[:6]
            tracked.append({
                "id": int(track_id),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(conf)
            })
        return tracked

def crop_with_padding(frame, bbox, pad=20):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(w, x2 + pad)
    ry2 = min(h, y2 + pad)
    return frame[ry1:ry2, rx1:rx2], rx1, ry1

SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]

def draw_skeleton(frame, keypoints, rx1, ry1):
    if keypoints is None or len(keypoints) < 17:
        return
    for i, j in SKELETON:
        kp1 = keypoints[i]
        kp2 = keypoints[j]
        if kp1[2] > 0.3 and kp2[2] > 0.3:
            pt1 = (int(kp1[0] + rx1), int(kp1[1] + ry1))
            pt2 = (int(kp2[0] + rx1), int(kp2[1] + ry1))
            cv2.line(frame, pt1, pt2, (255, 0, 255), 2)
    for kp in keypoints:
        if kp[2] > 0.3:
            pt = (int(kp[0] + rx1), int(kp[1] + ry1))
            cv2.circle(frame, pt, 4, (0, 255, 255), -1)

def process_video(source_path, output_path=None, display=False):
    if output_path is None:
        base, _ = os.path.splitext(source_path)
        output_path = f"{base}_marked.mp4"

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Halve inference rate: run YOLO + RTMPose every other frame
    infer_every_n = 2   # skip 1 frame between inferences
    print(f"Video FPS: {fps:.1f} | Running PeopleNet+RTMPose at every {infer_every_n} frames (~{fps/infer_every_n:.1f} inferences/sec)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    tracker = OCSortTracker(iou_threshold=0.25, max_lost=60, min_confidence=0.25)
    
    total_processing_time = 0.0
    tracked = []
    
    # Person State Managers
    track_history = {}    # track_id -> list of keypoint dicts
    track_states = {}     # track_id -> action dict

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Run YOLO + OC-SORT only on inference frames; reuse tracked on skipped frames
        if frame_count % infer_every_n == 0:
            dets = detect_persons_robust(frame)
            tracked = tracker.update(dets.tolist() if len(dets) > 0 else [], frame)

        # Map current active tracks
        active_ids = set()
        marked = frame.copy()
        
        # --- RTMPose + Action Classifier (also halved) ---
        for track in tracked:
            track_id = track["id"]
            active_ids.add(track_id)
            bbox = track["bbox"]

            if track_id not in track_history:
                track_history[track_id] = []
            if track_id not in track_states:
                track_states[track_id] = {"action": "STANDING", "conf": 0.5}

            # Run RTMPose only on inference frames
            if frame_count % infer_every_n == 0:
                roi, rx1, ry1 = crop_with_padding(frame, bbox, pad=20)
                if roi.size > 0:
                    keypoints, scores = rtmpose.infer(roi)
                    bbox_height = bbox[3] - bbox[1]

                    track_history[track_id].append({
                        "frame": frame_count,
                        "kps": keypoints,
                        "rx1": rx1,
                        "ry1": ry1,
                        "conf": scores,
                    })

                    # keep rolling 30 frames
                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)

                    action, action_conf = classify_action(keypoints, track_history[track_id], bbox_height)
                    track_states[track_id] = {"action": action, "conf": action_conf}
            
        # Draw active tracks
        ACTION_COLORS = {
            "STANDING":  (0, 255, 0),    # Green
            "SITTING":   (0, 255, 255),  # Yellow
            "LYING DOWN": (0, 0, 255),  # Red
            "FIGHTING":  (0, 128, 255),  # Orange
            "UNKNOWN":   (128, 128, 128),
        }
        for track in tracked:
            x1, y1, x2, y2 = map(int, track["bbox"])
            track_id = track["id"]
            state_info = track_states.get(track_id, {"action": "STANDING", "conf": 0.5})
            action = state_info["action"]
            action_conf = state_info["conf"]

            color = ACTION_COLORS.get(action, (0, 255, 0))
            det_conf = track["score"]
            label = f"ID:{track_id} {action} ({action_conf:.2f})"
            cv2.rectangle(marked, (x1, y1), (x2, y2), color, 2)
            cv2.putText(marked, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if len(track_history[track_id]) > 0:
                last_kps = track_history[track_id][-1].get("kps")
                last_rx1 = track_history[track_id][-1].get("rx1", 0)
                last_ry1 = track_history[track_id][-1].get("ry1", 0)
                draw_skeleton(marked, last_kps, last_rx1, last_ry1)

        count_label = f"Active Persons: {len(tracked)}"
        cv2.putText(marked, count_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        writer.write(marked)
        total_processing_time += (time.time() - start_time)
        frame_count += 1

        if display:
            cv2.imshow("Processed Video", marked)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    avg_latency = total_processing_time / frame_count if frame_count > 0 else 0
    print(f"Saved processed video to {output_path} ({frame_count} frames)")
    print(f"Avg latency per frame: {avg_latency:.4f} seconds")

def split_image_into_quadrants(image):
    h, w = image.shape[:2]
    mid_x, mid_y = w // 2, h // 2
    quadrants = [
        (0, 0, mid_x, mid_y, image[0:mid_y, 0:mid_x]),
        (mid_x, 0, w - mid_x, mid_y, image[0:mid_y, mid_x:w]),
        (0, mid_y, mid_x, h - mid_y, image[mid_y:h, 0:mid_x]),
        (mid_x, mid_y, w - mid_x, h - mid_y, image[mid_y:h, mid_x:w]),
    ]
    return quadrants

def stitch_quadrants(quadrants, image_shape):
    h, w = image_shape[:2]
    stitched = np.zeros(image_shape, dtype=np.uint8)
    for x, y, width, height, quad in quadrants:
        stitched[y:y + height, x:x + width] = quad
    return stitched

def process_image_with_quadrants(source_path, output_path):
    image = cv2.imread(source_path)
    if image is None:
        raise ValueError(f"Unable to read image: {source_path}")

    quadrants = split_image_into_quadrants(image)
    marked_quads = []

    for x, y, width, height, quad in quadrants:
        print(f"Processing quadrant at ({x}, {y}) for {source_path}...")
        dets = detect_persons_robust(quad)
        marked_quad = quad.copy()
        for d in dets:
            x1, y1, x2, y2, conf = d
            cv2.rectangle(marked_quad, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(marked_quad, f"{conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        marked_quads.append((x, y, width, height, marked_quad))

    stitched = stitch_quadrants(marked_quads, image.shape)
    cv2.imwrite(output_path, stitched)
    print(f"Saved stitched image {output_path}")

def process_images(input_dir="images", output_dir="marked_images", split_to_quadrants=False):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        abs_root = os.path.abspath(root)
        dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(abs_root, d)) != output_dir]
        if abs_root == output_dir or abs_root.startswith(output_dir + os.sep):
            continue

        rel_root = os.path.relpath(root, input_dir)
        target_root = os.path.join(output_dir, rel_root) if rel_root != "." else output_dir
        os.makedirs(target_root, exist_ok=True)

        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in valid_exts:
                continue

            source_path = os.path.join(root, file_name)
            output_path = os.path.join(target_root, file_name)
            print(f"Processing {source_path}...")

            start_time = time.time()

            if split_to_quadrants:
                process_image_with_quadrants(source_path, output_path)
            else:
                image = cv2.imread(source_path)
                if image is not None:
                    dets = detect_persons_robust(image)
                    marked = image.copy()
                    for d in dets:
                        x1, y1, x2, y2, conf = d
                        cv2.rectangle(marked, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(marked, f"{conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imwrite(output_path, marked)
            
            end_time = time.time()
            latency = end_time - start_time
            print(f"Saved {output_path}")
            print(f"Latency: {latency:.4f} seconds")

def process_videos(input_dir="videos", output_dir="marked_videos", display=False):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".ts", ".flv"}

    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        abs_root = os.path.abspath(root)
        dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(abs_root, d)) != output_dir]
        if abs_root == output_dir or abs_root.startswith(output_dir + os.sep):
            continue

        rel_root = os.path.relpath(root, input_dir)
        target_root = os.path.join(output_dir, rel_root) if rel_root != "." else output_dir
        os.makedirs(target_root, exist_ok=True)

        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in video_exts:
                continue

            source_path = os.path.join(root, file_name)
            base_name = os.path.splitext(file_name)[0]
            output_file_name = f"{base_name}_marked.mp4"
            output_path = os.path.join(target_root, output_file_name)
            print(f"Processing video {source_path}...")
            process_video(source_path, output_path=output_path, display=display)

if __name__ == "__main__":
    choice = input("Process images, videos, or a single video? [images/videos/video]: ").strip().lower()
    if choice == "video":
        source_path = input("Enter video path: ").strip()
        output_path = input("Enter output video path (leave blank to auto-generate): ").strip() or None
        process_video(source_path, output_path=output_path)
    elif choice == "videos":
        input_dir = input("Enter input video folder path: ").strip() or "videos"
        output_dir = input("Enter output video folder path: ").strip() or "marked_videos"
        display_choice = input("Display each video while processing? [y/N]: ").strip().lower()
        display_flag = display_choice in {"y", "yes"}

        if not os.path.isdir(input_dir):
            raise ValueError(f"Input folder does not exist: {input_dir}")

        process_videos(input_dir=input_dir, output_dir=output_dir, display=display_flag)
    else:
        input_dir = input("Enter input image folder path: ").strip() or "images"
        output_dir = input("Enter output image folder path: ").strip() or "marked_images"
        split_choice = input("Split each image into quadrants? [y/N]: ").strip().lower()
        split_flag = split_choice in {"y", "yes"}

        if not os.path.isdir(input_dir):
            raise ValueError(f"Input folder does not exist: {input_dir}")

        process_images(input_dir=input_dir, output_dir=output_dir, split_to_quadrants=split_flag)
