"""
Unified CV Pipeline Server
===========================
Merges pipelinePrototype.py (YOLO) and PeopleNetProto.py (PeopleNet ONNX)
into a single FastAPI backend with WebSocket live-relay streaming.

Supports:
  - Model selection (YOLO variants + PeopleNet)
  - Single-stream and multi-stream (up to 4) modes
  - Emulated live relay from video files at native FPS
  - Adaptive frame-skip to maintain ≥15 FPS per stream
  - REST API for control + WebSocket for frame streaming

Run:
  python server.py
"""

import os
import sys
import time
import json
import glob
import base64
import asyncio
import threading
import traceback
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision

# ---------------------------------------------------------------------------
# Fix ORT DLL loading on Windows (PyTorch bundles cuDNN)
# ---------------------------------------------------------------------------
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if torch_lib not in os.environ.get("PATH", ""):
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[server] Using device: {DEVICE}")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIRS = [
    os.path.join(PROJECT_ROOT, "Video_samples"),
    os.path.join(PROJECT_ROOT, "Internship project resource videos"),
    os.path.join(PROJECT_ROOT, "Pose_Samples"),
]

# ============================================================================
#  CV Pipeline Components (merged & deduplicated from both prototypes)
# ============================================================================

# ---------------------------------------------------------------------------
# PeopleNet Detector (from PeopleNetProto.py)
# ---------------------------------------------------------------------------
class PeopleNetDetector:
    """PeopleNet ResNet34 INT8 ONNX detector — replaces YOLO for person detection."""
    PERSON_CLASS = 0
    STRIDE       = 16
    INPUT_W      = 960
    INPUT_H      = 544
    SCALE        = 1.0 / 255.0

    def __init__(self, model_path="_/resnet34_peoplenet_int8.onnx"):
        self.model_path = os.path.join(PROJECT_ROOT, model_path)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"]
        )
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.session = ort.InferenceSession(
                self.model_path, sess_options=sess_opts, providers=providers
            )
        except Exception as e:
            print(f"[PeopleNet] CUDA provider failed ({e}). Falling back to CPU.")
            self.session = ort.InferenceSession(
                self.model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
            )
        self.input_name = self.session.get_inputs()[0].name
        out_names = [o.name for o in self.session.get_outputs()]
        out_shapes = [self.session.get_outputs()[i].shape for i in range(len(out_names))]
        self._cov_idx, self._bbox_idx = self._resolve_output_indices(out_shapes)
        self._lock = threading.Lock()
        print(f"[PeopleNet] Loaded | outputs: {out_names} | providers: {self.session.get_providers()}")

    @staticmethod
    def _resolve_output_indices(shapes):
        for i, s in enumerate(shapes):
            if len(s) == 4 and s[1] in (3, 12):
                if s[1] == 3:
                    return i, 1 - i
        return 0, 1

    def preprocess(self, bgr_frame):
        shape = bgr_frame.shape[:2]
        r = min(self.INPUT_W / shape[1], self.INPUT_H / shape[0])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.INPUT_W - new_unpad[0], self.INPUT_H - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            resized = cv2.resize(bgr_frame, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = bgr_frame
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # In-place conversion to avoid creating multiple large temporary arrays
        blob = padded.astype(np.float32)
        blob *= self.SCALE  # in-place multiply — no extra copy
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis, ...])
        return blob, r, dw, dh

    def detect(self, bgr_frame, conf_threshold=0.4, nms_iou=0.45):
        orig_h, orig_w = bgr_frame.shape[:2]
        blob, r, dw, dh = self.preprocess(bgr_frame)
        with self._lock:
            outputs = self.session.run(None, {self.input_name: blob})
        cov_map = outputs[self._cov_idx]
        bbox_map = outputs[self._bbox_idx]

        person_scores = cov_map[0, self.PERSON_CLASS]
        c = self.PERSON_CLASS
        bx1_map = bbox_map[0, c * 4 + 0]
        by1_map = bbox_map[0, c * 4 + 1]
        bx2_map = bbox_map[0, c * 4 + 2]
        by2_map = bbox_map[0, c * 4 + 3]

        grid_h, grid_w = person_scores.shape
        gy_idx, gx_idx = np.where(person_scores > conf_threshold)

        if len(gy_idx) == 0:
            return np.empty((0, 5))

        NORM = 35.0
        L = bx1_map[gy_idx, gx_idx]
        T = by1_map[gy_idx, gx_idx]
        R = bx2_map[gy_idx, gx_idx]
        B = by2_map[gy_idx, gx_idx]
        scores = person_scores[gy_idx, gx_idx]

        cx = gx_idx * self.STRIDE + self.STRIDE / 2.0
        cy = gy_idx * self.STRIDE + self.STRIDE / 2.0

        x1s = cx - L * NORM
        y1s = cy - T * NORM
        x2s = cx + R * NORM
        y2s = cy + B * NORM

        x1s = np.clip(x1s, 0, self.INPUT_W)
        y1s = np.clip(y1s, 0, self.INPUT_H)
        x2s = np.clip(x2s, 0, self.INPUT_W)
        y2s = np.clip(y2s, 0, self.INPUT_H)

        x1s = (x1s - dw) / r
        x2s = (x2s - dw) / r
        y1s = (y1s - dh) / r
        y2s = (y2s - dh) / r

        valid = (x2s > x1s) & (y2s > y1s)
        if not np.any(valid):
            return np.empty((0, 5))

        boxes_t = torch.from_numpy(
            np.stack([x1s[valid], y1s[valid], x2s[valid], y2s[valid]], axis=1).astype(np.float32)
        )
        scores_t = torch.from_numpy(scores[valid].astype(np.float32))
        keep = torchvision.ops.nms(boxes_t, scores_t, nms_iou)

        kept_boxes = boxes_t[keep].numpy()
        kept_scores = scores_t[keep].numpy().reshape(-1, 1)
        return np.concatenate([kept_boxes, kept_scores], axis=1)


# ---------------------------------------------------------------------------
# RTMPose Wrapper (shared by both pipelines)
# ---------------------------------------------------------------------------
class RTMPoseWrapper:
    def __init__(self, model_path="rtmpose-s.onnx"):
        self.model_path = os.path.join(PROJECT_ROOT, model_path)
        self.session = None
        self._lock = threading.Lock()
        if os.path.exists(self.model_path):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            try:
                self.session = ort.InferenceSession(self.model_path, providers=providers)
            except Exception as e:
                print(f"[RTMPose] CUDA provider failed ({e}). Falling back to CPU.")
                self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            print(f"[RTMPose] Loaded from {self.model_path}")
        else:
            print(f"[RTMPose] Warning: {self.model_path} not found. Using dummy keypoints.")

    def infer(self, roi):
        if self.session is None:
            h, w = roi.shape[:2]
            dummy_kps = np.zeros((17, 3))
            dummy_kps[:, 0] = w / 2
            dummy_kps[:, 1] = h / 2
            dummy_kps[:, 2] = 0.9
            dummy_kps[5, 1] = h * 0.2
            dummy_kps[6, 1] = h * 0.2
            dummy_kps[11, 1] = h * 0.6
            dummy_kps[12, 1] = h * 0.6
            return dummy_kps, np.ones(17) * 0.9

        h, w = roi.shape[:2]
        input_size = (192, 256)
        resized = cv2.resize(roi, input_size)
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0).astype(np.float32)

        input_name = self.session.get_inputs()[0].name
        with self._lock:
            outputs = self.session.run(None, {input_name: img})
        simcc_x, simcc_y = outputs[0][0], outputs[1][0]

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


# ---------------------------------------------------------------------------
# OC-SORT Tracker (shared)
# ---------------------------------------------------------------------------
from boxmot.trackers.bbox.ocsort.ocsort import OcSort

class OCSortTracker:
    def __init__(self, iou_threshold=0.25, max_lost=60, min_confidence=0.25):
        try:
            self.tracker = OcSort(
                det_thresh=min_confidence,
                max_age=max_lost,
                min_hits=2,
                iou_threshold=iou_threshold,
                delta_t=3,
                asso_func="iou",
                inertia=0.2,
                per_class=False
            )
        except TypeError:
            print("[OC-SORT] Strict parameters failed, falling back to safe kwargs")
            self.tracker = OcSort(
                det_thresh=min_confidence,
                max_age=max_lost,
                min_hits=2,
                iou_threshold=iou_threshold,
                per_class=False
            )

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
            x1, y1, x2, y2, track_id, conf, cls, ind = r
            tracked.append({
                "id": int(track_id),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(conf)
            })
        return tracked


# ---------------------------------------------------------------------------
# Action Classifier (shared)
# ---------------------------------------------------------------------------
def classify_action(keypoints, keypoint_history, bbox_height):
    """
    Classifies person action: STANDING, SITTING, FIGHTING, or LYING DOWN
    based on RTMPose 17-keypoint output (COCO format).
    """
    if keypoints is None or len(keypoints) < 17:
        return "UNKNOWN", 0.0

    def visible(idx, thresh=0.25):
        return keypoints[idx][2] > thresh

    shoulders_vis = visible(5) and visible(6)
    hips_vis      = visible(11) and visible(12)
    knees_vis     = visible(13) and visible(14)
    ankles_vis    = visible(15) and visible(16)

    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2 if shoulders_vis else None
    hip_y      = (keypoints[11][1] + keypoints[12][1]) / 2 if hips_vis else None
    knee_y     = (keypoints[13][1] + keypoints[14][1]) / 2 if knees_vis else None
    ankle_y    = (keypoints[15][1] + keypoints[16][1]) / 2 if ankles_vis else None

    bh = bbox_height if bbox_height > 1 else 1

    # LYING DOWN
    if shoulders_vis and hips_vis:
        vertical_delta = abs(shoulder_y - hip_y) / bh
        if vertical_delta < 0.24:
            return "LYING DOWN", min(1.0, 1.0 - vertical_delta / 0.24)

    # SITTING
    if shoulders_vis and hips_vis and knees_vis:
        hip_to_shoulder = (hip_y - shoulder_y) / bh
        knee_to_hip = (hip_y - knee_y) / bh
        if hip_to_shoulder > 0.15 and knee_to_hip > -0.05:
            if ankle_y is not None:
                knee_ankle_gap = (ankle_y - knee_y) / bh
                if knee_ankle_gap < 0.25:
                    return "SITTING", 0.75
            else:
                return "SITTING", 0.65

    # FIGHTING
    arms_raised = False
    if visible(7) and visible(8) and shoulders_vis:
        elbow_y = (keypoints[7][1] + keypoints[8][1]) / 1.5
        if elbow_y < shoulder_y:
            arms_raised = True
    if not arms_raised and (visible(9) or visible(10)):
        wrist_ys = []
        if visible(9): wrist_ys.append(keypoints[9][1])
        if visible(10): wrist_ys.append(keypoints[10][1])
        if shoulder_y is not None and any(w < shoulder_y for w in wrist_ys):
            arms_raised = True

    if arms_raised and len(keypoint_history) >= 5:
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

    # STANDING (default)
    if shoulders_vis and hips_vis:
        upright_ratio = (hip_y - shoulder_y) / bh
        if upright_ratio > 0.15:
            return "STANDING", min(1.0, upright_ratio / 0.5)

    return "STANDING", 0.5


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]

ACTION_COLORS = {
    "STANDING":   (0, 255, 0),
    "SITTING":    (0, 255, 255),
    "LYING DOWN": (0, 0, 255),
    "FIGHTING":   (0, 128, 255),
    "UNKNOWN":    (128, 128, 128),
}

def crop_with_padding(frame, bbox, pad=20):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(w, x2 + pad)
    ry2 = min(h, y2 + pad)
    return frame[ry1:ry2, rx1:rx2], rx1, ry1

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

def draw_annotations(frame, tracked, track_states, track_history):
    """Draw bounding boxes, labels, and skeletons on a frame."""
    for track in tracked:
        x1, y1, x2, y2 = map(int, track["bbox"])
        track_id = track["id"]
        state_info = track_states.get(track_id, {"action": "STANDING", "conf": 0.5})
        action = state_info["action"]
        action_conf = state_info["conf"]

        color = ACTION_COLORS.get(action, (0, 255, 0))
        label = f"ID:{track_id} {action} ({action_conf:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if track_id in track_history and len(track_history[track_id]) > 0:
            last = track_history[track_id][-1]
            draw_skeleton(frame, last.get("kps"), last.get("rx1", 0), last.get("ry1", 0))

    count_label = f"Active Persons: {len(tracked)}"
    cv2.putText(frame, count_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)


# ============================================================================
#  Model Registry
# ============================================================================
class ModelType(Enum):
    YOLO = "yolo"
    PEOPLENET = "peoplenet"

# Map of model_name -> (type, file_or_path)
MODEL_REGISTRY = {}

def _discover_models():
    """Discover available YOLO .pt files and PeopleNet ONNX in project root."""
    global MODEL_REGISTRY
    MODEL_REGISTRY = {}

    # Discover all .pt model files in project root
    for pt_file in glob.glob(os.path.join(PROJECT_ROOT, "*.pt")):
        name = os.path.splitext(os.path.basename(pt_file))[0]
        MODEL_REGISTRY[name] = (ModelType.YOLO, pt_file)

    # PeopleNet
    peoplenet_path = os.path.join(PROJECT_ROOT, "_", "resnet34_peoplenet_int8.onnx")
    if os.path.exists(peoplenet_path):
        MODEL_REGISTRY["peoplenet"] = (ModelType.PEOPLENET, peoplenet_path)

    print(f"[Models] Discovered: {list(MODEL_REGISTRY.keys())}")

_discover_models()

# Cached model instances (lazy-loaded)
_loaded_models = {}
_model_lock = threading.Lock()

def get_model(model_name: str):
    """Get or load a model instance by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    with _model_lock:
        if model_name in _loaded_models:
            return _loaded_models[model_name]

        model_type, model_path = MODEL_REGISTRY[model_name]
        if model_type == ModelType.YOLO:
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.to(DEVICE)
            _loaded_models[model_name] = ("yolo", model)
            print(f"[Models] Loaded YOLO model: {model_name}")
        elif model_type == ModelType.PEOPLENET:
            model = PeopleNetDetector(model_path)
            _loaded_models[model_name] = ("peoplenet", model)
            print(f"[Models] Loaded PeopleNet model")

        return _loaded_models[model_name]


# Limit concurrent inference calls to prevent memory pressure crashes
_inference_semaphore = threading.Semaphore(2)

def detect_persons(frame, model_name: str):
    """Unified person detection — dispatches to YOLO or PeopleNet."""
    model_type, model_instance = get_model(model_name)

    with _inference_semaphore:
        if model_type == "yolo":
            # Dynamically find which class IDs correspond to "person"
            target_classes = []
            for idx, name in model_instance.names.items():
                name_lower = name.lower()
                if "person" in name_lower or "lying" in name_lower or "sitting" in name_lower:
                    target_classes.append(idx)
            if not target_classes:
                target_classes = [0] # fallback

            res = model_instance.predict(frame, classes=target_classes, device=DEVICE, verbose=False, conf=0.2, imgsz=640)[0]
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy().reshape(-1, 1)
            # Add class ID as the 6th column if tracking wants to know it (optional, but good practice)
            # Currently tracker expects (N, 5), so we leave it as [x1, y1, x2, y2, conf]
            if len(boxes) > 0:
                return np.concatenate([boxes, confs], axis=1)
            return np.empty((0, 5))
        elif model_type == "peoplenet":
            return model_instance.detect(frame, conf_threshold=0.4, nms_iou=0.45)


# ============================================================================
#  Video Discovery
# ============================================================================
def discover_videos():
    """Find all video files in known directories."""
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".ts", ".flv"}
    videos = []

    for vdir in VIDEO_DIRS:
        if not os.path.isdir(vdir):
            continue
        for root, dirs, files in os.walk(vdir):
            # Skip marked_* output directories
            dirs[:] = [d for d in dirs if not d.startswith("marked_")]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in video_exts:
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, PROJECT_ROOT)
                    videos.append({
                        "name": f,
                        "path": rel_path.replace("\\", "/"),
                        "full_path": full_path,
                    })
    return videos


# ============================================================================
#  RTMPose singleton
# ============================================================================
rtmpose = RTMPoseWrapper()


# ============================================================================
#  Stream Processor — one per video stream
# ============================================================================
class StreamProcessor:
    """
    Processes a single video stream in a background thread.
    Emulates live relay by reading frames at the video's native FPS.
    Adaptively adjusts inference frequency to maintain ≥15 FPS output.
    """

    def __init__(self, stream_id: str, video_path: str, model_name: str, frame_callback, target_fps=15):
        self.stream_id = stream_id
        self.video_path = video_path
        self.model_name = model_name
        self.frame_callback = frame_callback
        self.target_fps = target_fps

        self._running = False
        self._reader_thread = None
        self._processor_thread = None
        self._loop = None

        self.current_fps = 0.0
        self.person_count = 0
        self.frame_number = 0
        self.total_frames = 0
        
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()

    def start(self, loop):
        if self._running:
            return
        self._running = True
        self._loop = loop
        self._reader_thread = threading.Thread(target=self._read_frames, daemon=True, name=f"read-{self.stream_id}")
        self._processor_thread = threading.Thread(target=self._process_frames, daemon=True, name=f"proc-{self.stream_id}")
        self._reader_thread.start()
        self._processor_thread.start()

    def stop(self):
        self._running = False
        self.new_frame_event.set()
        if self._reader_thread:
            self._reader_thread.join(timeout=2)
            self._reader_thread = None
        if self._processor_thread:
            self._processor_thread.join(timeout=2)
            self._processor_thread = None

    def _read_frames(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self._send_error(f"Cannot open video: {self.video_path}")
                return

            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = 1.0 / native_fps

            print(f"[Stream {self.stream_id}] Reader started — {self.video_path} @ {native_fps:.1f} FPS")

            while self._running:
                loop_start = time.time()
                ret, frame = cap.read()
                
                if not ret:
                    if self.video_path.startswith(("rtsp://", "http://", "https://")):
                        self._send_error("Stream ended or disconnected.")
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                h, w = frame.shape[:2]
                if h > 720 or w > 1280:
                    scale = min(1280.0 / w, 720.0 / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h))

                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()

                if not self.video_path.startswith(("rtsp://", "http://", "https://")):
                    total_elapsed = time.time() - loop_start
                    sleep_time = frame_interval - total_elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            cap.release()
            print(f"[Stream {self.stream_id}] Reader stopped")
        except Exception as e:
            traceback.print_exc()
            self._send_error(str(e))

    def _process_frames(self):
        import base64
        try:
            tracker = OCSortTracker(iou_threshold=0.25, max_lost=60, min_confidence=0.25)
            track_history = {}
            track_states = {}
            tracked = []

            fps_window = []
            target_interval = 1.0 / self.target_fps
            print(f"[Stream {self.stream_id}] Processor started")

            while self._running:
                loop_start = time.time()
                
                if not self.new_frame_event.wait(timeout=1.0):
                    continue
                
                with self.frame_lock:
                    frame = self.latest_frame
                self.new_frame_event.clear()
                
                if frame is None:
                    continue

                process_start = time.time()
                
                dets = detect_persons(frame, self.model_name)
                tracked = tracker.update(dets.tolist() if len(dets) > 0 else [], frame)

                active_ids = set()
                rtmpose_count = 0 
                
                for track in tracked:
                    track_id = track["id"]
                    active_ids.add(track_id)
                    bbox = track["bbox"]

                    if track_id not in track_history:
                        track_history[track_id] = []
                    if track_id not in track_states:
                        track_states[track_id] = {"action": "STANDING", "conf": 0.5, "last_rtm_frame": 0}

                    state = track_states[track_id]
                    if self.frame_number - state.get("last_rtm_frame", 0) >= 3 and rtmpose_count < 2:
                        roi, rx1, ry1 = crop_with_padding(frame, bbox, pad=20)
                        if roi.size > 0:
                            keypoints, scores = rtmpose.infer(roi)
                            rtmpose_count += 1
                            bbox_height = bbox[3] - bbox[1]
                            track_history[track_id].append({
                                "frame": self.frame_number,
                                "kps": keypoints,
                                "rx1": rx1,
                                "ry1": ry1,
                                "conf": scores,
                            })
                            if len(track_history[track_id]) > 30:
                                track_history[track_id].pop(0)
                            action, action_conf = classify_action(keypoints, track_history[track_id], bbox_height)
                            state["action"] = action
                            state["conf"] = action_conf
                            state["last_rtm_frame"] = self.frame_number

                stale = [tid for tid in track_history if tid not in active_ids]
                for tid in stale:
                    if len(track_history.get(tid, [])) > 0:
                        last_frame = track_history[tid][-1].get("frame", 0)
                        if self.frame_number - last_frame > 120:
                            del track_history[tid]
                            track_states.pop(tid, None)
                    else:
                        del track_history[tid]
                        track_states.pop(tid, None)

                marked = frame.copy()
                draw_annotations(marked, tracked, track_states, track_history)

                model_label = f"Model: {self.model_name.upper()}"
                cv2.putText(marked, model_label, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
                _, buffer = cv2.imencode('.jpg', marked, encode_params)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')

                tracks_info = []
                for track in tracked:
                    tid = track["id"]
                    state = track_states.get(tid, {"action": "STANDING", "conf": 0.5})
                    tracks_info.append({
                        "id": tid,
                        "bbox": track["bbox"],
                        "action": state["action"],
                        "conf": round(state["conf"], 2),
                    })

                fps_window.append(time.time())
                if len(fps_window) > 30:
                    fps_window.pop(0)
                if len(fps_window) >= 2:
                    elapsed = fps_window[-1] - fps_window[0]
                    self.current_fps = (len(fps_window) - 1) / elapsed if elapsed > 0 else 0
                else:
                    self.current_fps = 0

                self.person_count = len(tracked)
                self.frame_number += 1

                frame_data = {
                    "type": "frame",
                    "stream_id": self.stream_id,
                    "frame": frame_b64,
                    "fps": round(self.current_fps, 1),
                    "person_count": self.person_count,
                    "tracks": tracks_info,
                    "frame_number": self.frame_number,
                    "total_frames": self.total_frames,
                    "model": self.model_name,
                    "infer_every_n": 1, 
                }
                self._send_frame(frame_data)

                process_elapsed = time.time() - loop_start
                sleep_time = target_interval - process_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f"[Stream {self.stream_id}] Processor stopped")
        except Exception as e:
            traceback.print_exc()
            self._send_error(str(e))

    def _send_frame(self, data: dict):
        """Thread-safe: schedule frame delivery on the asyncio event loop."""
        if self._loop and self._running:
            if getattr(self, '_frame_queued', False):
                return  # Drop frame to prevent memory leak!
            self._frame_queued = True
            
            async def wrapped():
                try:
                    await self.frame_callback(data)
                finally:
                    self._frame_queued = False
                    
            asyncio.run_coroutine_threadsafe(wrapped(), self._loop)

    def _send_error(self, message: str):
        if self._loop:
            error_data = {"type": "error", "stream_id": self.stream_id, "message": message}
            asyncio.run_coroutine_threadsafe(self.frame_callback(error_data), self._loop)


# ============================================================================
#  FastAPI Application
# ============================================================================
app = FastAPI(title="CV Pipeline Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_streams: dict[str, StreamProcessor] = {}
ws_clients: set[WebSocket] = set()
_streams_lock = threading.Lock()


async def broadcast_frame(data: dict):
    """Send frame data to all connected WebSocket clients."""
    global ws_clients
    dead_clients = set()
    for ws in ws_clients.copy():
        try:
            await ws.send_json(data)
        except Exception:
            dead_clients.add(ws)
    ws_clients -= dead_clients


# --- REST Endpoints ---

@app.get("/api/models")
async def list_models():
    """List available CV models."""
    models = []
    for name, (mtype, path) in MODEL_REGISTRY.items():
        models.append({
            "name": name,
            "type": mtype.value,
            "file": os.path.basename(path),
        })
    # Sort: peoplenet first, then by name
    models.sort(key=lambda m: (0 if m["name"] == "peoplenet" else 1, m["name"]))
    return {"models": models}


@app.get("/api/videos")
async def list_videos():
    """List available video files."""
    videos = discover_videos()
    return {"videos": videos}

class FolderRequest(BaseModel):
    path: str

@app.post("/api/folders")
async def add_folder(req: FolderRequest):
    """Add a new folder to the video discovery list."""
    if not os.path.isdir(req.path):
        return JSONResponse(status_code=400, content={"error": "Invalid folder path. Does not exist."})
        
    if req.path not in VIDEO_DIRS:
        VIDEO_DIRS.append(req.path)
        
    return {"message": "Folder added", "videos": discover_videos()}


@app.get("/api/status")
async def pipeline_status():
    """Get current pipeline status."""
    streams_info = {}
    with _streams_lock:
        for sid, sp in active_streams.items():
            streams_info[sid] = {
                "stream_id": sid,
                "video": sp.video_path,
                "model": sp.model_name,
                "fps": round(sp.current_fps, 1),
                "person_count": sp.person_count,
                "frame_number": sp.frame_number,
                "total_frames": sp.total_frames,
                "infer_every_n": sp.infer_every_n,
                "running": sp._running,
            }
    return {
        "running": len(active_streams) > 0,
        "stream_count": len(active_streams),
        "streams": streams_info,
        "ws_clients": len(ws_clients),
    }


@app.post("/api/start")
async def start_pipeline(config: dict):
    """
    Start the CV pipeline.
    Body: { "model": "yolo26n", "mode": "single"|"multi", "videos": ["path1", ...] }
    """
    model_name = config.get("model", "yolo26n")
    mode = config.get("mode", "single")
    video_paths = config.get("videos", [])

    if not video_paths:
        return JSONResponse(status_code=400, content={"error": "No videos specified"})

    if model_name not in MODEL_REGISTRY:
        return JSONResponse(status_code=400, content={
            "error": f"Unknown model: {model_name}",
            "available": list(MODEL_REGISTRY.keys())
        })

    # Stop existing streams
    await stop_pipeline()

    # Resolve video paths
    resolved = []
    for vp in video_paths:
        if vp.startswith(("rtsp://", "http://", "https://")):
            resolved.append(vp)
            continue
        full = os.path.join(PROJECT_ROOT, vp)
        if os.path.exists(full):
            resolved.append(full)
        elif os.path.exists(vp):
            resolved.append(vp)
        else:
            return JSONResponse(status_code=400, content={"error": f"Video not found: {vp}"})

    if mode == "single":
        resolved = resolved[:1]
    else:
        resolved = resolved[:4]  # max 4 streams

    # Pre-load the model (so first frame isn't slow)
    try:
        get_model(model_name)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to load model: {e}"})

    # Set target FPS based on mode
    target_fps = 25.0 if mode == "single" else 15.0

    loop = asyncio.get_event_loop()

    with _streams_lock:
        for i, vpath in enumerate(resolved):
            sid = f"stream_{i}"
            sp = StreamProcessor(
                stream_id=sid,
                video_path=vpath,
                model_name=model_name,
                frame_callback=broadcast_frame,
                target_fps=target_fps,
            )
            active_streams[sid] = sp
            sp.start(loop)

    # Broadcast status
    await broadcast_frame({
        "type": "status",
        "message": f"Started {len(resolved)} stream(s) with {model_name}",
        "mode": mode,
        "stream_count": len(resolved),
    })

    return {
        "status": "started",
        "mode": mode,
        "model": model_name,
        "streams": len(resolved),
    }


@app.post("/api/stop")
async def stop_pipeline():
    """Stop all active streams."""
    with _streams_lock:
        for sid, sp in active_streams.items():
            sp.stop()
        active_streams.clear()

    await broadcast_frame({"type": "status", "message": "Pipeline stopped", "stream_count": 0})
    return {"status": "stopped"}


# --- WebSocket Endpoint ---

@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """WebSocket endpoint for live frame streaming."""
    await ws.accept()
    ws_clients.add(ws)
    print(f"[WS] Client connected. Total: {len(ws_clients)}")

    try:
        # Send initial status
        await ws.send_json({
            "type": "status",
            "message": "Connected to CV Pipeline Server",
            "stream_count": len(active_streams),
            "models": list(MODEL_REGISTRY.keys()),
        })

        # Keep connection alive — listen for control messages
        while True:
            try:
                data = await ws.receive_text()
                msg = json.loads(data)
                # Client can send ping/pong or control messages
                if msg.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    finally:
        ws_clients.discard(ws)
        print(f"[WS] Client disconnected. Total: {len(ws_clients)}")


# ============================================================================
#  Entry Point
# ============================================================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  CV Pipeline Dashboard Server")
    print(f"  Models: {list(MODEL_REGISTRY.keys())}")
    print(f"  Videos: {len(discover_videos())} files found")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
