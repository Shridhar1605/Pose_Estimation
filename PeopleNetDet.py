"""
PeopleNetDet.py – PeopleNet-only inference script
===================================================
Model: resnet34_peoplenet_int8.onnx  (NVIDIA PeopleNet)

Input contract
--------------
  • Color order : RGB  (not BGR)
  • Spatial size : 960 × 544  (W × H)
  • Normalization: divide by 255.0  (float32, range 0..1)
  • Tensor layout: NCHW  (batch, channels, height, width)

Output contract
---------------
  PeopleNet's ONNX outputs are raw feature maps:
    output[0]  – confidence map  shape (1, num_classes, grid_h, grid_w)
    output[1]  – bounding-box map shape (1, num_classes*4, grid_h, grid_w)

  Decoding pipeline applied here:
    1. Squeeze batch dimension
    2. For every grid cell whose class score > CONF_THRESHOLD (0.3),
       reconstruct the bounding box in normalised [0,1] coordinates.
    3. Apply Non-Maximum Suppression (IoU threshold 0.45).
    4. Scale boxes back to the original image dimensions.

Usage
-----
  python PeopleNetDet.py                        # interactive prompts
  python PeopleNetDet.py --input path/to/file   # single image or video
"""

import os

# ── Cross-platform device / runtime selection (macOS MPS+CoreML, CUDA, CPU) ──
# platform_utils also forces UTF-8 console output on every OS.
from platform_utils import DEVICE as _PLATFORM_DEVICE, device_name, get_ort_providers, make_ort_session

import argparse
import sys

import cv2
import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# ── Constants ─────────────────────────────────────────────────────────────────
SCRIPT_DIR            = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH    = os.path.join(SCRIPT_DIR, "_", "resnet34_peoplenet_int8.onnx")
LABELS_PATH           = os.path.join(SCRIPT_DIR, "_", "labels.txt")

# PeopleNet fixed input dimensions (W, H)
PEOPLENET_INPUT_W = 960
PEOPLENET_INPUT_H = 544

# Detection thresholds
CONF_THRESHOLD = 0.3      # minimum class confidence to keep a cell
IOU_THRESHOLD  = 0.45     # NMS IoU threshold

# Supported image extensions
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Label names – loaded from labels.txt if present
LABEL_NAMES: list[str] = []
if os.path.isfile(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as _f:
        LABEL_NAMES = [line.strip() for line in _f if line.strip()]

# Colours per class (BGR for OpenCV drawing)
CLASS_COLORS = [
    (0, 255, 80),    # person  – green
    (80, 180, 255),  # bag     – light-blue
    (255, 80, 80),   # face    – red-ish
]


# ─────────────────────────────────────────────────────────────────────────────
# CUDA device activation
# ─────────────────────────────────────────────────────────────────────────────

def activate_cuda_device(device_index: int = 0) -> str:
    """
    Attempt to activate the requested CUDA GPU. Returns a device string such
    as "cuda:0" or falls back to "cpu" if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        cuda_built = getattr(torch.cuda, "is_built", None)
        if callable(cuda_built):
            cpu_only = not cuda_built()
        else:
            cpu_only = torch.version.cuda is None

        print(f"PyTorch version : {torch.__version__}")
        print(f"Built with CUDA : {torch.version.cuda}")
        if cpu_only:
            print("CUDA is not available – this PyTorch build is CPU-only.")
            print("Install a CUDA-enabled build, e.g.:")
            print("  pip install torch torchvision torchaudio "
                  "--index-url https://download.pytorch.org/whl/cu128")
        else:
            print("CUDA is not available in this environment.")
        return "cpu"

    device_count = torch.cuda.device_count()
    print(f"CUDA device count : {device_count}")
    if device_index >= device_count:
        print(f"Requested CUDA device {device_index} not found. Falling back to device 0.")
        device_index = 0

    device_str = f"cuda:{device_index}"
    try:
        torch.cuda.set_device(device_index)
        props = torch.cuda.get_device_properties(device_index)
        print(f"Setting device    : {device_str}  ({props.name})")
        dummy = torch.zeros((1,), device=device_str)
        dummy.add_(1)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"Activated device  : {device_str}")
        return device_str
    except Exception as exc:
        print(f"Warning: could not activate {device_str}: {exc}")
        return "cpu"


DEVICE = activate_cuda_device(0) if _PLATFORM_DEVICE.startswith("cuda") else _PLATFORM_DEVICE
print(f"Using device      : {DEVICE} ({device_name()})\n")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_peoplenet_model(model_path: str | None = None) -> "ort.InferenceSession":
    """Load the PeopleNet ONNX model and return an ONNXRuntime session."""
    if ort is None:
        raise ImportError(
            "ONNXRuntime is required. Install with:\n"
            "  pip install onnxruntime-gpu   # GPU\n"
            "  pip install onnxruntime       # CPU-only"
        )

    model_path = model_path or DEFAULT_MODEL_PATH
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"PeopleNet model not found: {model_path}\n"
            "Place resnet34_peoplenet_int8.onnx in the '_' sub-folder."
        )

    available = ort.get_available_providers()
    providers  = get_ort_providers()          # CUDA -> CoreML (macOS) -> CPU
    print(f"ORT providers available : {available}")
    print(f"ORT providers selected  : {providers}")

    session = make_ort_session(model_path, providers, log_prefix="[PeopleNet]")
    inp = session.get_inputs()[0]
    print(f"Model input  : {inp.name}  shape={inp.shape}  dtype={inp.type}")
    for out in session.get_outputs():
        print(f"Model output : {out.name}  shape={out.shape}  dtype={out.type}")
    return session


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Convert an OpenCV BGR frame to the PeopleNet input tensor:
      RGB → resize 960×544 → /255.0 → NCHW float32
    Returns array of shape (1, 3, 544, 960).
    """
    # BGR → RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize to model input (W=960, H=544)
    resized = cv2.resize(rgb, (PEOPLENET_INPUT_W, PEOPLENET_INPUT_H))
    # Normalise to [0, 1]
    tensor = resized.astype(np.float32) / 255.0
    # HWC → CHW, then add batch dimension → NCHW
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Non-Maximum Suppression
# ─────────────────────────────────────────────────────────────────────────────

def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = IOU_THRESHOLD,
) -> np.ndarray:
    """
    CPU NMS.  boxes: (N, 4) in [x1, y1, x2, y2] normalised coords.
    Returns an array of kept indices.
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1  = np.maximum(x1[i], x1[rest])
        yy1  = np.maximum(y1[i], y1[rest])
        xx2  = np.minimum(x2[i], x2[rest])
        yy2  = np.minimum(y2[i], y2[rest])
        inter   = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union   = areas[i] + areas[rest] - inter
        iou     = np.where(union > 0, inter / union, 0.0)
        order   = order[1:][iou <= iou_threshold]

    return np.array(keep, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Output decoding
# ─────────────────────────────────────────────────────────────────────────────

def decode_peoplenet_output(
    conf_map: np.ndarray,
    bbox_map: np.ndarray,
    conf_threshold: float = CONF_THRESHOLD,
    iou_threshold:  float = IOU_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode PeopleNet raw feature-map outputs.

    Parameters
    ----------
    conf_map : shape (num_classes, grid_h, grid_w)
        Per-cell class confidence scores (already sigmoid-activated by the model).
    bbox_map : shape (num_classes*4, grid_h, grid_w)
        Per-cell bbox offsets [dx1, dy1, dx2, dy2] relative to the grid cell.
    conf_threshold : keep cells with score > this value.
    iou_threshold  : NMS overlap threshold.

    Returns
    -------
    boxes  : (K, 4)  float32  normalised [x1, y1, x2, y2]
    scores : (K,)    float32
    labels : (K,)    int32
    """
    if conf_map.ndim != 3:
        raise ValueError(f"conf_map must be 3-D (classes, H, W), got {conf_map.shape}")
    if bbox_map.ndim != 3:
        raise ValueError(f"bbox_map must be 3-D (classes*4, H, W), got {bbox_map.shape}")

    num_classes, grid_h, grid_w = conf_map.shape
    # PeopleNet bbox_map layout: (num_classes * 4, grid_h, grid_w)
    # Class cls_idx occupies channels [cls_idx*4 .. cls_idx*4 + 3]
    # i.e. exactly one set of (dx1, dy1, dx2, dy2) offsets per class per cell.
    expected_bbox_channels = num_classes * 4
    if bbox_map.shape[0] != expected_bbox_channels:
        raise ValueError(
            f"bbox_map has {bbox_map.shape[0]} channels, expected {expected_bbox_channels} "
            f"(num_classes={num_classes} * 4)"
        )

    boxes_list:  list[list[float]] = []
    scores_list: list[float]       = []
    labels_list: list[int]         = []

    for cls_idx in range(num_classes):
        score_map = conf_map[cls_idx]                        # (grid_h, grid_w)
        active    = np.argwhere(score_map > conf_threshold)  # shape (M, 2)
        if active.size == 0:
            continue

        base = cls_idx * 4  # channel offset in bbox_map for this class

        # NVIDIA DetectNet_v2 offset scale factor
        OFFSET_SCALE = 35.0
        
        # Grid cell dimensions in pixels (model input size / grid size)
        stride_x = PEOPLENET_INPUT_W / grid_w
        stride_y = PEOPLENET_INPUT_H / grid_h

        for r, c in active:
            score = float(score_map[r, c])

            # DetectNet v2 offsets are relative to the grid cell top-left,
            # scaled by OFFSET_SCALE.
            dx1 = float(bbox_map[base + 0, r, c])
            dy1 = float(bbox_map[base + 1, r, c])
            dx2 = float(bbox_map[base + 2, r, c])
            dy2 = float(bbox_map[base + 3, r, c])

            # Calculate absolute pixel coordinates in the 960x544 space
            x1_px = (c * stride_x) - (dx1 * OFFSET_SCALE)
            y1_px = (r * stride_y) - (dy1 * OFFSET_SCALE)
            x2_px = (c * stride_x) + (dx2 * OFFSET_SCALE)
            y2_px = (r * stride_y) + (dy2 * OFFSET_SCALE)

            # Normalise to [0, 1] and clip
            x1 = max(0.0, min(1.0, x1_px / PEOPLENET_INPUT_W))
            y1 = max(0.0, min(1.0, y1_px / PEOPLENET_INPUT_H))
            x2 = max(0.0, min(1.0, x2_px / PEOPLENET_INPUT_W))
            y2 = max(0.0, min(1.0, y2_px / PEOPLENET_INPUT_H))

            if x2 <= x1 or y2 <= y1:
                continue  # degenerate box

            boxes_list.append([x1, y1, x2, y2])
            scores_list.append(score)
            labels_list.append(cls_idx)

    if not boxes_list:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,),   dtype=np.float32),
            np.empty((0,),   dtype=np.int32),
        )

    boxes  = np.array(boxes_list,  dtype=np.float32)
    scores = np.array(scores_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)

    keep   = non_max_suppression(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], labels[keep]


# ─────────────────────────────────────────────────────────────────────────────
# Inference on a single frame
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    session: "ort.InferenceSession",
    frame_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run PeopleNet on one BGR frame.

    Returns
    -------
    boxes  : (K, 4)  pixel coords [x1, y1, x2, y2] in the *original* frame size
    scores : (K,)    confidence values
    labels : (K,)    class indices  (0=person, 1=bag, 2=face)
    """
    orig_h, orig_w = frame_bgr.shape[:2]
    input_tensor   = preprocess_frame(frame_bgr)

    input_name = session.get_inputs()[0].name
    raw_output = session.run(None, {input_name: input_tensor})

    if len(raw_output) < 2:
        raise RuntimeError(
            f"Expected at least 2 output tensors from PeopleNet, got {len(raw_output)}."
        )

    # Squeeze batch dimension from each output
    conf_map = np.squeeze(raw_output[0])   # (num_classes, grid_h, grid_w)
    bbox_map = np.squeeze(raw_output[1])   # (num_classes*4, grid_h, grid_w)

    boxes_norm, scores, labels = decode_peoplenet_output(conf_map, bbox_map)

    # Scale normalised boxes back to original pixel dimensions
    if boxes_norm.shape[0] > 0:
        boxes_px = boxes_norm.copy()
        boxes_px[:, [0, 2]] *= orig_w   # x1, x2
        boxes_px[:, [1, 3]] *= orig_h   # y1, y2
        boxes_px = np.clip(boxes_px, 0, None)
    else:
        boxes_px = boxes_norm

    return boxes_px, scores, labels


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_detections(
    frame_bgr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the frame."""
    out = frame_bgr.copy()
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
        score    = float(scores[idx])
        cls_idx  = int(labels[idx])
        color    = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]
        cls_name = (
            LABEL_NAMES[cls_idx]
            if LABEL_NAMES and cls_idx < len(LABEL_NAMES)
            else f"class_{cls_idx}"
        )
        label_text = f"{cls_name} {score:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        # Background rectangle for legible text
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, ty - th - baseline), (x1 + tw, ty + baseline), color, -1)
        cv2.putText(
            out, label_text,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 1, cv2.LINE_AA,
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Output path helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_output_path(input_path: str, suffix: str = "_peoplenet") -> str:
    """
    Build an output file path in the *same folder* as the input,
    appending *suffix* before the file extension.
    e.g.  /data/crowd.jpg  →  /data/crowd_peoplenet.jpg
    """
    folder    = os.path.dirname(os.path.abspath(input_path))
    base, ext = os.path.splitext(os.path.basename(input_path))
    return os.path.join(folder, f"{base}{suffix}{ext}")


# ─────────────────────────────────────────────────────────────────────────────
# Image processing
# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    input_path: str,
    session: "ort.InferenceSession",
    output_path: str | None = None,
) -> str:
    """
    Run PeopleNet on a single image and save the annotated result.

    Returns the path to the saved output image.
    """
    frame = cv2.imread(input_path)
    if frame is None:
        raise ValueError(f"Cannot read image: {input_path}")

    output_path = output_path or make_output_path(input_path)

    print(f"  Processing image : {input_path}")
    boxes, scores, labels = run_inference(session, frame)
    print(f"  Detections       : {len(boxes)}")

    annotated = draw_detections(frame, boxes, scores, labels)
    cv2.imwrite(output_path, annotated)
    print(f"  Saved output     : {output_path}")
    return output_path


def process_image_directory(
    input_dir: str,
    session: "ort.InferenceSession",
) -> None:
    """Walk *input_dir* and process every supported image file in-place."""
    processed = 0
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() not in VALID_IMAGE_EXTS:
                continue
            src = os.path.join(root, fname)
            try:
                process_image(src, session)
                processed += 1
            except Exception as exc:
                print(f"  [ERROR] {src}: {exc}")
    print(f"\nDone. Processed {processed} image(s) from '{input_dir}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Video processing
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
    input_path: str,
    session: "ort.InferenceSession",
    output_path: str | None = None,
    display: bool = False,
) -> str:
    """
    Run PeopleNet on every frame of a video and write the annotated video
    to *output_path* (defaults to same folder as input with _peoplenet suffix).

    Returns the path to the saved output video.
    """
    output_path = output_path or make_output_path(input_path, suffix="_peoplenet")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    print(f"  Processing video : {input_path}")
    print(f"  Resolution       : {orig_w}×{orig_h}  @{fps:.1f} fps  ({n_frames} frames)")
    print(f"  Output           : {output_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores, labels = run_inference(session, frame)
        annotated = draw_detections(frame, boxes, scores, labels)
        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{n_frames}  detections: {len(boxes)}")

        if display:
            cv2.imshow("PeopleNet – press Q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    print(f"  Done. {frame_idx} frame(s) written to '{output_path}'.")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {
        ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"
    }


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VALID_IMAGE_EXTS


def interactive_main(session: "ort.InferenceSession") -> None:
    """Simple interactive prompt when no CLI arguments are provided."""
    print("\n-- PeopleNet Detection ----------------------------------------------")
    print("Provide a path to an image file, a video file, or an image directory.")
    input_path = input("Input path: ").strip().strip('"').strip("'")

    if not input_path:
        print("No input provided. Exiting.")
        return

    if os.path.isdir(input_path):
        process_image_directory(input_path, session)
    elif os.path.isfile(input_path):
        if _is_video(input_path):
            process_video(input_path, session)
        elif _is_image(input_path):
            process_image(input_path, session)
        else:
            print(f"Unsupported file type: {input_path}")
    else:
        print(f"Path not found: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PeopleNet ONNX inference – person, bag, and face detection."
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Path to an image file, video file, or directory of images.",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the PeopleNet ONNX model (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Output path. Defaults to the same folder as the input with a "
            "'_peoplenet' suffix added to the filename."
        ),
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=CONF_THRESHOLD,
        help=f"Confidence threshold (default: {CONF_THRESHOLD}).",
    )
    parser.add_argument(
        "--display", "-d",
        action="store_true",
        help="Show live preview window while processing a video.",
    )
    args = parser.parse_args()

    session = load_peoplenet_model(args.model)

    if args.input is None:
        # No CLI argument → fall back to interactive prompt
        interactive_main(session)
        return

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"[ERROR] Path not found: {input_path}")
        sys.exit(1)

    if os.path.isdir(input_path):
        process_image_directory(input_path, session)
    elif _is_video(input_path):
        process_video(input_path, session, output_path=args.output, display=args.display)
    elif _is_image(input_path):
        process_image(input_path, session, output_path=args.output)
    else:
        print(f"[ERROR] Unsupported file type: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
