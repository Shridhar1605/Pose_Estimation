"""
Unified CV Pipeline Server
===========================
The single pipeline that supports BOTH detectors:

  * YOLO26 (Ultralytics .pt weights, any file matching *.pt in the project root)
  * NVIDIA PeopleNet (ResNet34 INT8 ONNX, _/resnet34_peoplenet_int8.onnx)

followed by the shared stages: OC-SORT tracking -> RTMPose keypoints -> rule
based action classification (STANDING / SITTING / LYING DOWN / FIGHTING).

Cross-platform: runs on macOS (Apple Silicon: YOLO on Metal/MPS, ONNX models on
CoreML), Linux/Windows (CUDA) and plain CPU. Device selection lives in
platform_utils.py.

Endpoints
---------
  GET  /api/system            runtime + hardware summary
  GET  /api/models            discovered models
  GET  /api/videos            discovered video sources
  POST /api/folders           add a folder to scan for videos   {"path": "..."}
  POST /api/upload            upload a video file (multipart)
  GET  /api/status            live status of running streams
  POST /api/start             {"model": "...", "mode": "single|multi", "videos": [...]}
  POST /api/stop
  WS   /ws/stream             JSON frames: {type:"frame", stream_id, frame(b64 jpeg), fps, tracks, ...}
  GET  /                      the built dashboard (dashboard/dist) when present

Run:
  .venv/bin/python server.py            # http://localhost:8000
"""
from __future__ import annotations

# platform_utils must be imported before torch / onnxruntime / cv2
from platform_utils import (  # noqa: E402
    DEVICE, IS_MAC, device_name, get_ort_providers, make_ocsort, make_ort_session,
    system_summary,
)

import asyncio
import base64
import glob
import json
import os
import platform
import re
import shutil
import threading
import time
import traceback
from collections import Counter, deque
from enum import Enum
from typing import Optional

import logging

import cv2
import numpy as np
import torch
import torchvision

logging.getLogger("boxmot").setLevel(logging.WARNING)  # silence per-tracker INFO banners

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
print(f"[server] torch device: {DEVICE} ({device_name()}) | ORT providers: {get_ort_providers()}")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "Video_samples", "uploads")
VIDEO_DIRS = [
    os.path.join(PROJECT_ROOT, "Video_samples"),
    os.path.join(PROJECT_ROOT, "videos"),
    os.path.join(PROJECT_ROOT, "Pose_Samples"),
    os.path.join(PROJECT_ROOT, "Internship project resource videos"),
]
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".ts", ".flv", ".m4v"}
MAX_STREAMS = 4
STREAM_URL_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")

# ============================================================================
#  CV Pipeline Components
# ============================================================================

# ---------------------------------------------------------------------------
# PeopleNet Detector (DetectNet_v2 GridBox decoder)
# ---------------------------------------------------------------------------
class PeopleNetDetector:
    """NVIDIA PeopleNet ResNet34 INT8 ONNX detector (person / bag / face)."""
    PERSON_CLASS = 0
    STRIDE = 16
    INPUT_W = 960
    INPUT_H = 544
    SCALE = 1.0 / 255.0
    BBOX_NORM = 35.0  # DetectNet_v2 bbox scale

    def __init__(self, model_path="_/resnet34_peoplenet_int8.onnx"):
        self.model_path = model_path if os.path.isabs(model_path) else os.path.join(PROJECT_ROOT, model_path)
        self.session = make_ort_session(self.model_path, log_prefix="[PeopleNet]")
        self.input_name = self.session.get_inputs()[0].name
        out_shapes = [o.shape for o in self.session.get_outputs()]
        self._cov_idx, self._bbox_idx = self._resolve_output_indices(out_shapes)
        self._lock = threading.Lock()
        print(f"[PeopleNet] loaded {os.path.basename(self.model_path)} | providers: {self.session.get_providers()}")

    @staticmethod
    def _resolve_output_indices(shapes):
        for i, s in enumerate(shapes):
            if len(s) == 4 and s[1] == 3:
                return i, 1 - i
        return 0, 1

    def preprocess(self, bgr_frame):
        shape = bgr_frame.shape[:2]
        r = min(self.INPUT_W / shape[1], self.INPUT_H / shape[0])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (self.INPUT_W - new_unpad[0]) / 2, (self.INPUT_H - new_unpad[1]) / 2
        resized = cv2.resize(bgr_frame, new_unpad, interpolation=cv2.INTER_LINEAR) if shape[::-1] != new_unpad else bgr_frame
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        blob = padded.astype(np.float32)
        blob *= self.SCALE
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis, ...])
        return blob, r, dw, dh

    def detect(self, bgr_frame, conf_threshold=0.4, nms_iou=0.45):
        blob, r, dw, dh = self.preprocess(bgr_frame)
        with self._lock:
            outputs = self.session.run(None, {self.input_name: blob})
        cov_map, bbox_map = outputs[self._cov_idx], outputs[self._bbox_idx]

        c = self.PERSON_CLASS
        person_scores = cov_map[0, c]
        gy, gx = np.where(person_scores > conf_threshold)
        if len(gy) == 0:
            return np.empty((0, 5))

        L, T = bbox_map[0, c * 4 + 0][gy, gx], bbox_map[0, c * 4 + 1][gy, gx]
        R, B = bbox_map[0, c * 4 + 2][gy, gx], bbox_map[0, c * 4 + 3][gy, gx]
        scores = person_scores[gy, gx]
        cx = gx * self.STRIDE + self.STRIDE / 2.0
        cy = gy * self.STRIDE + self.STRIDE / 2.0

        x1 = np.clip(cx - L * self.BBOX_NORM, 0, self.INPUT_W)
        y1 = np.clip(cy - T * self.BBOX_NORM, 0, self.INPUT_H)
        x2 = np.clip(cx + R * self.BBOX_NORM, 0, self.INPUT_W)
        y2 = np.clip(cy + B * self.BBOX_NORM, 0, self.INPUT_H)
        x1, x2 = (x1 - dw) / r, (x2 - dw) / r
        y1, y2 = (y1 - dh) / r, (y2 - dh) / r

        valid = (x2 > x1) & (y2 > y1)
        if not np.any(valid):
            return np.empty((0, 5))
        boxes_t = torch.from_numpy(np.stack([x1[valid], y1[valid], x2[valid], y2[valid]], axis=1).astype(np.float32))
        scores_t = torch.from_numpy(scores[valid].astype(np.float32))
        keep = torchvision.ops.nms(boxes_t, scores_t, nms_iou)  # CPU tensors -> works on every platform
        return np.concatenate([boxes_t[keep].numpy(), scores_t[keep].numpy().reshape(-1, 1)], axis=1)


# ---------------------------------------------------------------------------
# RTMPose (SimCC) keypoint estimator
# ---------------------------------------------------------------------------
class RTMPoseWrapper:
    INPUT_W, INPUT_H = 192, 256
    MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __init__(self, model_path="rtmpose-s.onnx"):
        self.model_path = model_path if os.path.isabs(model_path) else os.path.join(PROJECT_ROOT, model_path)
        self.session = None
        self._lock = threading.Lock()
        if os.path.exists(self.model_path):
            self.session = make_ort_session(self.model_path, log_prefix="[RTMPose]")
            self.input_name = self.session.get_inputs()[0].name
            print(f"[RTMPose] loaded {os.path.basename(self.model_path)} | providers: {self.session.get_providers()}")
        else:
            print(f"[RTMPose] {self.model_path} not found -> action classifier will use placeholder keypoints")

    @property
    def available(self) -> bool:
        return self.session is not None

    def infer(self, roi):
        h, w = roi.shape[:2]
        if self.session is None:
            kps = np.zeros((17, 3))
            kps[:, 0], kps[:, 1], kps[:, 2] = w / 2, h / 2, 0.9
            kps[5, 1] = kps[6, 1] = h * 0.2
            kps[11, 1] = kps[12, 1] = h * 0.6
            return kps, np.ones(17) * 0.9

        img = cv2.cvtColor(cv2.resize(roi, (self.INPUT_W, self.INPUT_H)), cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - self.MEAN) / self.STD
        img = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32)
        with self._lock:
            simcc_x, simcc_y = self.session.run(None, {self.input_name: img})
        simcc_x, simcc_y = simcc_x[0], simcc_y[0]
        x_locs, y_locs = np.argmax(simcc_x, axis=1), np.argmax(simcc_y, axis=1)
        scores = (np.max(simcc_x, axis=1) + np.max(simcc_y, axis=1)) / 2
        kps = np.zeros((17, 3))
        kps[:, 0] = x_locs / (self.INPUT_W * 2) * w
        kps[:, 1] = y_locs / (self.INPUT_H * 2) * h
        kps[:, 2] = scores
        return kps, scores


# ---------------------------------------------------------------------------
# OC-SORT tracker wrapper
# ---------------------------------------------------------------------------
class OCSortTracker:
    def __init__(self, iou_threshold=0.25, max_lost=60, min_confidence=0.25):
        self.tracker = make_ocsort(iou_threshold, max_lost, min_confidence)

    def update(self, detections, frame=None):
        if len(detections) == 0:
            dets_np = np.empty((0, 6), dtype=np.float32)
        else:
            dets_np = np.asarray(detections, dtype=np.float32)
            dets_np = np.concatenate([dets_np, np.zeros((dets_np.shape[0], 1), dtype=np.float32)], axis=1)
        if frame is None:
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
        res = self.tracker.update(dets_np, frame)
        tracked = []
        for r in res:
            x1, y1, x2, y2, track_id, conf = r[:6]
            tracked.append({"id": int(track_id), "bbox": [float(x1), float(y1), float(x2), float(y2)], "score": float(conf)})
        return tracked


# ---------------------------------------------------------------------------
# Action classifier (COCO-17 keypoints, rule based)
# ---------------------------------------------------------------------------
def classify_action(keypoints, keypoint_history, bbox_height):
    if keypoints is None or len(keypoints) < 17:
        return "UNKNOWN", 0.0

    def visible(idx, thresh=0.25):
        return keypoints[idx][2] > thresh

    shoulders_vis = visible(5) and visible(6)
    hips_vis = visible(11) and visible(12)
    knees_vis = visible(13) and visible(14)
    ankles_vis = visible(15) and visible(16)
    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2 if shoulders_vis else None
    hip_y = (keypoints[11][1] + keypoints[12][1]) / 2 if hips_vis else None
    knee_y = (keypoints[13][1] + keypoints[14][1]) / 2 if knees_vis else None
    ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2 if ankles_vis else None
    bh = bbox_height if bbox_height > 1 else 1

    if shoulders_vis and hips_vis:
        vertical_delta = abs(shoulder_y - hip_y) / bh
        if vertical_delta < 0.24:
            return "LYING DOWN", min(1.0, 1.0 - vertical_delta / 0.24)

    if shoulders_vis and hips_vis and knees_vis:
        hip_to_shoulder = (hip_y - shoulder_y) / bh
        knee_to_hip = (hip_y - knee_y) / bh
        if hip_to_shoulder > 0.15 and knee_to_hip > -0.05:
            if ankle_y is not None:
                if (ankle_y - knee_y) / bh < 0.25:
                    return "SITTING", 0.75
            else:
                return "SITTING", 0.65

    arms_raised = False
    if visible(7) and visible(8) and shoulders_vis:
        if (keypoints[7][1] + keypoints[8][1]) / 1.5 < shoulder_y:
            arms_raised = True
    if not arms_raised and (visible(9) or visible(10)) and shoulder_y is not None:
        wrist_ys = [keypoints[i][1] for i in (9, 10) if visible(i)]
        if any(w < shoulder_y for w in wrist_ys):
            arms_raised = True

    if arms_raised and len(keypoint_history) >= 5:
        wrist_positions = []
        for entry in keypoint_history[-8:]:
            kps = entry.get("kps")
            if kps is not None and len(kps) >= 17:
                wrist_positions.append(((kps[9][0] + kps[10][0]) / 2, (kps[9][1] + kps[10][1]) / 2))
        if len(wrist_positions) >= 3:
            dists = [np.hypot(wrist_positions[i][0] - wrist_positions[i - 1][0],
                              wrist_positions[i][1] - wrist_positions[i - 1][1])
                     for i in range(1, len(wrist_positions))]
            avg_speed = sum(dists) / len(dists)
            if avg_speed > 6.0:
                return "FIGHTING", min(1.0, avg_speed / 25.0)

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
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]
ACTION_COLORS = {  # BGR - same colorblind-safe palette the dashboard uses (validated)
    "STANDING": (138, 166, 34),     # #22A68A teal
    "SITTING": (20, 138, 192),      # #C08A14 amber
    "LYING DOWN": (102, 67, 217),   # #D94366 rose
    "FIGHTING": (234, 113, 131),    # #8371EA violet
    "UNKNOWN": (132, 118, 107),     # #6B7684 slate
}


def crop_with_padding(frame, bbox, pad=20):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
    return frame[ry1:ry2, rx1:rx2], rx1, ry1


def draw_skeleton(frame, keypoints, rx1, ry1):
    if keypoints is None or len(keypoints) < 17:
        return
    for i, j in SKELETON:
        kp1, kp2 = keypoints[i], keypoints[j]
        if kp1[2] > 0.3 and kp2[2] > 0.3:
            cv2.line(frame, (int(kp1[0] + rx1), int(kp1[1] + ry1)), (int(kp2[0] + rx1), int(kp2[1] + ry1)), (255, 0, 255), 2)
    for kp in keypoints:
        if kp[2] > 0.3:
            cv2.circle(frame, (int(kp[0] + rx1), int(kp[1] + ry1)), 3, (0, 255, 255), -1)


def draw_annotations(frame, tracked, track_states, track_history, draw_pose=True):
    for track in tracked:
        x1, y1, x2, y2 = map(int, track["bbox"])
        tid = track["id"]
        state = track_states.get(tid, {"action": "STANDING", "conf": 0.5})
        color = ACTION_COLORS.get(state["action"], (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"#{tid} {state['action']} {state['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(th + 6, y1)
        lx = min(max(0, x1), frame.shape[1] - tw - 6)  # keep label inside the frame
        cv2.rectangle(frame, (lx, ly - th - 6), (lx + tw + 6, ly), color, -1)
        cv2.putText(frame, label, (lx + 3, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)
        if draw_pose and track_history.get(tid):
            last = track_history[tid][-1]
            draw_skeleton(frame, last.get("kps"), last.get("rx1", 0), last.get("ry1", 0))


# ============================================================================
#  Model registry
# ============================================================================
class ModelType(Enum):
    YOLO = "yolo"
    PEOPLENET = "peoplenet"


MODEL_REGISTRY: dict[str, tuple[ModelType, str]] = {}
PEOPLENET_PATH = os.path.join(PROJECT_ROOT, "_", "resnet34_peoplenet_int8.onnx")


def _discover_models():
    MODEL_REGISTRY.clear()
    for pt_file in sorted(glob.glob(os.path.join(PROJECT_ROOT, "*.pt"))):
        MODEL_REGISTRY[os.path.splitext(os.path.basename(pt_file))[0]] = (ModelType.YOLO, pt_file)
    if os.path.exists(PEOPLENET_PATH):
        MODEL_REGISTRY["peoplenet"] = (ModelType.PEOPLENET, PEOPLENET_PATH)
    print(f"[Models] discovered: {list(MODEL_REGISTRY.keys())}")


_discover_models()

_loaded_models: dict[str, tuple[str, object]] = {}
_model_lock = threading.Lock()
_yolo_locks: dict[str, threading.Lock] = {}
_inference_semaphore = threading.Semaphore(2)


def get_model(model_name: str):
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
            # warm-up so the first live frame is not slow (MPS compiles kernels lazily)
            model.predict(np.zeros((640, 640, 3), dtype=np.uint8), device=DEVICE, verbose=False, imgsz=640)
            _yolo_locks[model_name] = threading.Lock()
            _loaded_models[model_name] = ("yolo", model)
            print(f"[Models] loaded YOLO '{model_name}' on {DEVICE} | classes: {model.names}")
        else:
            _loaded_models[model_name] = ("peoplenet", PeopleNetDetector(model_path))
        return _loaded_models[model_name]


def _person_classes(model) -> list[int]:
    ids = [i for i, n in model.names.items() if any(k in n.lower() for k in ("person", "lying", "sitting", "standing"))]
    return ids or [0]


def detect_persons(frame, model_name: str):
    """Unified detection -> ndarray (N, 5) [x1, y1, x2, y2, conf]."""
    model_type, model = get_model(model_name)
    with _inference_semaphore:
        if model_type == "yolo":
            # MPS is not safe for concurrent predicts on one model -> serialize per model
            with _yolo_locks[model_name]:
                res = model.predict(frame, classes=_person_classes(model), device=DEVICE,
                                    verbose=False, conf=0.25, imgsz=640)[0]
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy().reshape(-1, 1)
            return np.concatenate([boxes, confs], axis=1) if len(boxes) else np.empty((0, 5))
        return model.detect(frame, conf_threshold=0.4, nms_iou=0.45)


def model_info(name: str) -> dict:
    mtype, path = MODEL_REGISTRY[name]
    info = {"name": name, "type": mtype.value, "file": os.path.basename(path),
            "size_mb": round(os.path.getsize(path) / 1e6, 1)}
    if mtype == ModelType.YOLO:
        info["backend"] = f"PyTorch / {DEVICE}"
        info["description"] = "Ultralytics YOLO26 (end-to-end, NMS-free) person detector"
    else:
        info["backend"] = "ONNX Runtime / " + get_ort_providers()[0].replace("ExecutionProvider", "")
        info["description"] = "NVIDIA PeopleNet ResNet34 INT8 (DetectNet_v2 GridBox, 960x544)"
    if name in _loaded_models and mtype == ModelType.YOLO:
        info["classes"] = list(_loaded_models[name][1].names.values())
    return info


# ============================================================================
#  Video discovery
# ============================================================================
def _source_kind(path: str) -> str:
    if path.startswith(STREAM_URL_PREFIXES):
        return "stream"
    if re.fullmatch(r"(webcam:)?\d+", path):
        return "webcam"
    return "file"


def _video_meta(full_path: str) -> dict:
    try:
        cap = cv2.VideoCapture(full_path)
        meta = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(cap.get(cv2.CAP_PROP_FPS) or 0, 1),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        meta["duration_s"] = round(meta["frames"] / meta["fps"], 1) if meta["fps"] else None
        return meta
    except Exception:
        return {}


_meta_cache: dict[str, dict] = {}


def discover_videos():
    videos = []
    seen = set()
    for vdir in VIDEO_DIRS:
        if not os.path.isdir(vdir):
            continue
        for root, dirs, files in os.walk(vdir):
            dirs[:] = [d for d in dirs if not d.startswith("marked_")]
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() not in VIDEO_EXTS:
                    continue
                full = os.path.join(root, f)
                if full in seen:
                    continue
                seen.add(full)
                rel = os.path.relpath(full, PROJECT_ROOT).replace(os.sep, "/") if full.startswith(PROJECT_ROOT) else full
                key = (full, os.path.getmtime(full))
                if key not in _meta_cache:
                    _meta_cache[key] = _video_meta(full)
                videos.append({"name": f, "path": rel, "full_path": full,
                               "folder": os.path.relpath(root, PROJECT_ROOT).replace(os.sep, "/") if root.startswith(PROJECT_ROOT) else root,
                               **_meta_cache[key]})
    return videos


def resolve_source(vp: str):
    """Return (opencv_source, display_name) or raise FileNotFoundError."""
    kind = _source_kind(vp)
    if kind == "stream":
        return vp, vp
    if kind == "webcam":
        return int(vp.split(":")[-1]), f"Webcam {vp.split(':')[-1]}"
    for candidate in (os.path.join(PROJECT_ROOT, vp), vp, os.path.expanduser(vp)):
        if os.path.isfile(candidate):
            return candidate, os.path.basename(candidate)
    raise FileNotFoundError(vp)


# ============================================================================
#  RTMPose singleton
# ============================================================================
rtmpose = RTMPoseWrapper()


# ============================================================================
#  Stream processor: one per video source
# ============================================================================
class StreamProcessor:
    """Reads a source at native FPS in one thread and runs the pipeline in another.
    Always processes the *latest* frame so the output stays live; slow hardware
    simply lowers the effective inference FPS rather than building a backlog."""

    def __init__(self, stream_id: str, source, display_name: str, model_name: str,
                 frame_callback, target_fps=15.0, jpeg_quality=70):
        self.stream_id = stream_id
        self.source = source
        self.display_name = display_name
        self.model_name = model_name
        self.frame_callback = frame_callback
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality

        self._running = False
        self._reader_thread = None
        self._processor_thread = None
        self._loop = None
        self._frame_queued = False

        self.current_fps = 0.0
        self.latency_ms = 0.0
        self.person_count = 0
        self.frame_number = 0
        self.total_frames = 0
        self.native_fps = 0.0
        self.source_frame_idx = 0
        self.action_counts: dict[str, int] = {}
        self.unique_ids: set[int] = set()
        self.started_at = time.time()
        self.last_error: Optional[str] = None

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()

    @property
    def is_live(self) -> bool:
        return isinstance(self.source, int) or str(self.source).startswith(STREAM_URL_PREFIXES)

    # -- lifecycle ---------------------------------------------------------
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
        for t in (self._reader_thread, self._processor_thread):
            if t:
                t.join(timeout=3)
        self._reader_thread = self._processor_thread = None

    # -- reader ------------------------------------------------------------
    def _open_capture(self):
        if isinstance(self.source, int) and IS_MAC:
            return cv2.VideoCapture(self.source, cv2.CAP_AVFOUNDATION)
        return cv2.VideoCapture(self.source)

    def _read_frames(self):
        try:
            cap = self._open_capture()
            if not cap.isOpened():
                self._send_error(f"Cannot open source: {self.display_name}")
                return
            self.native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_live else 0
            frame_interval = 1.0 / self.native_fps
            print(f"[{self.stream_id}] reader: {self.display_name} @ {self.native_fps:.1f} fps, {self.total_frames} frames")

            while self._running:
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    if self.is_live:
                        self._send_error("Stream ended or disconnected.")
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop file sources
                    self.source_frame_idx = 0
                    continue
                self.source_frame_idx += 1

                h, w = frame.shape[:2]
                if h > 720 or w > 1280:
                    scale = min(1280.0 / w, 720.0 / h)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()

                if not self.is_live:
                    sleep_time = frame_interval - (time.time() - loop_start)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            cap.release()
            print(f"[{self.stream_id}] reader stopped")
        except Exception as e:
            traceback.print_exc()
            self._send_error(str(e))

    # -- processor ---------------------------------------------------------
    def _process_frames(self):
        try:
            tracker = OCSortTracker(iou_threshold=0.25, max_lost=60, min_confidence=0.25)
            track_history: dict[int, list] = {}
            track_states: dict[int, dict] = {}
            fps_window: deque = deque(maxlen=30)
            target_interval = 1.0 / self.target_fps
            print(f"[{self.stream_id}] processor started ({self.model_name})")

            while self._running:
                loop_start = time.time()
                if not self.new_frame_event.wait(timeout=1.0):
                    continue
                with self.frame_lock:
                    frame = self.latest_frame
                self.new_frame_event.clear()
                if frame is None:
                    continue

                t0 = time.time()
                dets = detect_persons(frame, self.model_name)
                tracked = tracker.update(dets.tolist() if len(dets) else [], frame)

                active_ids = set()
                pose_budget = 3
                for track in tracked:
                    tid = track["id"]
                    active_ids.add(tid)
                    self.unique_ids.add(tid)
                    bbox = track["bbox"]
                    track_history.setdefault(tid, [])
                    state = track_states.setdefault(tid, {"action": "STANDING", "conf": 0.5, "last_rtm_frame": -10})
                    if self.frame_number - state["last_rtm_frame"] >= 3 and pose_budget > 0:
                        roi, rx1, ry1 = crop_with_padding(frame, bbox, pad=20)
                        if roi.size > 0 and roi.shape[0] > 8 and roi.shape[1] > 8:
                            keypoints, scores = rtmpose.infer(roi)
                            pose_budget -= 1
                            track_history[tid].append({"frame": self.frame_number, "kps": keypoints, "rx1": rx1, "ry1": ry1, "conf": scores})
                            if len(track_history[tid]) > 30:
                                track_history[tid].pop(0)
                            action, action_conf = classify_action(keypoints, track_history[tid], bbox[3] - bbox[1])
                            state.update(action=action, conf=action_conf, last_rtm_frame=self.frame_number)

                for tid in [t for t in track_history if t not in active_ids]:
                    hist = track_history.get(tid) or []
                    if not hist or self.frame_number - hist[-1].get("frame", 0) > 120:
                        track_history.pop(tid, None)
                        track_states.pop(tid, None)

                self.latency_ms = (time.time() - t0) * 1000.0

                marked = frame.copy()
                draw_annotations(marked, tracked, track_states, track_history, draw_pose=rtmpose.available)
                _, buffer = cv2.imencode(".jpg", marked, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                frame_b64 = base64.b64encode(buffer).decode("ascii")

                tracks_info = []
                for track in tracked:
                    tid = track["id"]
                    st = track_states.get(tid, {"action": "STANDING", "conf": 0.5})
                    tracks_info.append({"id": tid, "bbox": [round(v, 1) for v in track["bbox"]],
                                        "score": round(track["score"], 2), "action": st["action"], "conf": round(st["conf"], 2)})
                self.action_counts = dict(Counter(t["action"] for t in tracks_info))

                fps_window.append(time.time())
                if len(fps_window) >= 2:
                    elapsed = fps_window[-1] - fps_window[0]
                    self.current_fps = (len(fps_window) - 1) / elapsed if elapsed > 0 else 0.0
                self.person_count = len(tracked)
                self.frame_number += 1

                self._send_frame({
                    "type": "frame",
                    "stream_id": self.stream_id,
                    "source": self.display_name,
                    "frame": frame_b64,
                    "width": marked.shape[1],
                    "height": marked.shape[0],
                    "fps": round(self.current_fps, 1),
                    "latency_ms": round(self.latency_ms, 1),
                    "person_count": self.person_count,
                    "unique_persons": len(self.unique_ids),
                    "actions": self.action_counts,
                    "tracks": tracks_info,
                    "frame_number": self.frame_number,
                    "source_frame": self.source_frame_idx,
                    "total_frames": self.total_frames,
                    "native_fps": round(self.native_fps, 1),
                    "model": self.model_name,
                    "device": DEVICE if MODEL_REGISTRY[self.model_name][0] == ModelType.YOLO else get_ort_providers()[0],
                    "ts": time.time(),
                })

                sleep_time = target_interval - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            print(f"[{self.stream_id}] processor stopped")
        except Exception as e:
            traceback.print_exc()
            self._send_error(str(e))

    # -- delivery ----------------------------------------------------------
    def _send_frame(self, data: dict):
        if not (self._loop and self._running):
            return
        if self._frame_queued:
            return  # drop frame: never let a slow client build a backlog
        self._frame_queued = True

        async def wrapped():
            try:
                await self.frame_callback(data)
            finally:
                self._frame_queued = False

        try:
            asyncio.run_coroutine_threadsafe(wrapped(), self._loop)
        except RuntimeError:
            self._frame_queued = False

    def _send_error(self, message: str):
        self.last_error = message
        if self._loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.frame_callback({"type": "error", "stream_id": self.stream_id, "message": message}), self._loop)
            except RuntimeError:
                pass

    def status(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "source": self.display_name,
            "model": self.model_name,
            "fps": round(self.current_fps, 1),
            "latency_ms": round(self.latency_ms, 1),
            "person_count": self.person_count,
            "unique_persons": len(self.unique_ids),
            "actions": self.action_counts,
            "frame_number": self.frame_number,
            "total_frames": self.total_frames,
            "native_fps": round(self.native_fps, 1),
            "uptime_s": round(time.time() - self.started_at, 1),
            "running": self._running,
            "error": self.last_error,
        }


# ============================================================================
#  FastAPI application
# ============================================================================
app = FastAPI(title="Unified CV Pipeline", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

active_streams: dict[str, StreamProcessor] = {}
ws_clients: set[WebSocket] = set()
_streams_lock = threading.Lock()
_pipeline_started_at: Optional[float] = None
_current_config: dict = {}


async def broadcast(data: dict):
    dead = set()
    for ws in ws_clients.copy():
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    for ws in dead:
        ws_clients.discard(ws)


# --- REST -------------------------------------------------------------------
@app.get("/api/system")
async def api_system():
    info = system_summary()
    info.update({
        "rtmpose_available": rtmpose.available,
        "models": [model_info(n) for n in MODEL_REGISTRY],
        "video_dirs": VIDEO_DIRS,
        "max_streams": MAX_STREAMS,
    })
    return info


@app.get("/api/models")
async def api_models():
    models = [model_info(n) for n in MODEL_REGISTRY]
    models.sort(key=lambda m: (0 if m["name"] == "peoplenet" else 1, m["name"]))
    return {"models": models}


@app.get("/api/videos")
async def api_videos():
    return {"videos": discover_videos()}


class FolderRequest(BaseModel):
    path: str


@app.post("/api/folders")
async def api_add_folder(req: FolderRequest):
    path = os.path.expanduser(req.path)
    if not os.path.isdir(path):
        return JSONResponse(status_code=400, content={"error": f"Folder does not exist: {req.path}"})
    if path not in VIDEO_DIRS:
        VIDEO_DIRS.append(path)
    return {"message": "Folder added", "videos": discover_videos()}


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in VIDEO_EXTS:
        return JSONResponse(status_code=400, content={"error": f"Unsupported file type: {ext or 'unknown'}"})
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(file.filename))
    dest = os.path.join(UPLOAD_DIR, safe)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    rel = os.path.relpath(dest, PROJECT_ROOT).replace(os.sep, "/")
    return {"message": "Uploaded", "path": rel, "videos": discover_videos()}


@app.get("/api/status")
async def api_status():
    with _streams_lock:
        streams = {sid: sp.status() for sid, sp in active_streams.items()}
    return {
        "running": len(streams) > 0,
        "stream_count": len(streams),
        "streams": streams,
        "ws_clients": len(ws_clients),
        "config": _current_config,
        "uptime_s": round(time.time() - _pipeline_started_at, 1) if _pipeline_started_at else 0,
        "device": DEVICE,
    }


@app.post("/api/start")
async def api_start(config: dict):
    global _pipeline_started_at, _current_config
    model_name = config.get("model") or next(iter(MODEL_REGISTRY), None)
    mode = config.get("mode", "single")
    video_paths = [v for v in config.get("videos", []) if v]
    if not video_paths:
        return JSONResponse(status_code=400, content={"error": "No video sources specified"})
    if model_name not in MODEL_REGISTRY:
        return JSONResponse(status_code=400, content={"error": f"Unknown model: {model_name}", "available": list(MODEL_REGISTRY)})

    await api_stop()

    resolved = []
    for vp in video_paths[: (1 if mode == "single" else MAX_STREAMS)]:
        try:
            resolved.append(resolve_source(vp))
        except FileNotFoundError:
            return JSONResponse(status_code=400, content={"error": f"Video not found: {vp}"})

    try:
        await asyncio.get_running_loop().run_in_executor(None, get_model, model_name)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Failed to load model: {e}"})

    target_fps = float(config.get("target_fps") or (25.0 if mode == "single" else 15.0))
    jpeg_quality = int(config.get("jpeg_quality") or (75 if mode == "single" else 60))
    loop = asyncio.get_running_loop()

    with _streams_lock:
        for i, (src, name) in enumerate(resolved):
            sid = f"stream_{i}"
            sp = StreamProcessor(sid, src, name, model_name, broadcast, target_fps=target_fps, jpeg_quality=jpeg_quality)
            active_streams[sid] = sp
            sp.start(loop)
    _pipeline_started_at = time.time()
    _current_config = {"model": model_name, "mode": mode, "videos": video_paths[: len(resolved)], "target_fps": target_fps}

    await broadcast({"type": "status", "status": "started", "message": f"Started {len(resolved)} stream(s) with {model_name}",
                     "mode": mode, "model": model_name, "stream_count": len(resolved),
                     "streams": [{"stream_id": f"stream_{i}", "source": n} for i, (_, n) in enumerate(resolved)]})
    return {"status": "started", "mode": mode, "model": model_name, "streams": len(resolved)}


@app.post("/api/stop")
async def api_stop():
    global _pipeline_started_at, _current_config
    with _streams_lock:
        procs = list(active_streams.values())
        active_streams.clear()
    if procs:
        await asyncio.get_running_loop().run_in_executor(None, lambda: [p.stop() for p in procs])
    _pipeline_started_at = None
    _current_config = {}
    await broadcast({"type": "status", "status": "stopped", "message": "Pipeline stopped", "stream_count": 0})
    return {"status": "stopped"}


# --- WebSocket --------------------------------------------------------------
@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    print(f"[WS] client connected ({len(ws_clients)} total)")
    try:
        await ws.send_json({"type": "status", "status": "connected", "message": "Connected to CV Pipeline Server",
                            "stream_count": len(active_streams), "models": list(MODEL_REGISTRY), "device": DEVICE})
        while True:
            try:
                msg = json.loads(await ws.receive_text())
                if msg.get("type") == "ping":
                    await ws.send_json({"type": "pong", "ts": time.time()})
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    finally:
        ws_clients.discard(ws)
        print(f"[WS] client disconnected ({len(ws_clients)} total)")


# --- Static dashboard (built with `npm run build` in ./dashboard) ------------
DASHBOARD_DIST = os.path.join(PROJECT_ROOT, "dashboard", "dist")
if os.path.isfile(os.path.join(DASHBOARD_DIST, "index.html")):
    app.mount("/", StaticFiles(directory=DASHBOARD_DIST, html=True), name="dashboard")
    print(f"[server] serving dashboard from {DASHBOARD_DIST}")
else:
    @app.get("/")
    async def root():
        return {"message": "Unified CV Pipeline API. Build the dashboard (cd dashboard && npm run build) or run it with `npm run dev`.",
                "docs": "/docs"}


@app.on_event("shutdown")
async def _shutdown():
    await api_stop()


# ============================================================================
#  Entry point
# ============================================================================
if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    print(f"\n{'=' * 64}\n  Unified CV Pipeline Server\n"
          f"  Platform : {platform.system()} {platform.machine()}\n"
          f"  Device   : {DEVICE} ({device_name()})\n"
          f"  ORT      : {get_ort_providers()}\n"
          f"  Models   : {list(MODEL_REGISTRY)}\n"
          f"  Videos   : {len(discover_videos())} files\n"
          f"  URL      : http://localhost:{port}\n{'=' * 64}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")
