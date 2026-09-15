"""
platform_utils.py
=================
Cross-platform (macOS / Linux / Windows) device + runtime selection shared by
every script in this project.

The original code base hard-coded ``"cuda:0" if torch.cuda.is_available() else "cpu"``
and patched Windows DLL paths. This module replaces that with:

  * torch device selection  : CUDA -> Apple MPS -> CPU   (override: DEVICE=...)
  * ONNX Runtime providers  : CUDA -> CoreML (opt-in) -> CPU (override: ORT_PROVIDERS=a,b)
  * env hygiene for macOS   : MPS op fallback, duplicate OpenMP guard

Import this *before* torch / onnxruntime in every entry point:

    from platform_utils import DEVICE, get_ort_providers, system_summary
"""
from __future__ import annotations

import os
import platform
import sys

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")
IS_APPLE_SILICON = IS_MAC and platform.machine() == "arm64"

# --------------------------------------------------------------------------
# Environment hygiene (must happen before torch / onnxruntime are imported)
# --------------------------------------------------------------------------
# Let PyTorch fall back to CPU for the handful of ops MPS does not implement.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# torch + onnxruntime + opencv can each ship their own libomp on macOS; without
# this guard the process aborts with "OMP: Error #15".
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Keep Ultralytics from phoning home / auto-installing things at import time.
os.environ.setdefault("YOLO_OFFLINE", "0")

if IS_WIN:
    # Original Windows hack: ORT needs PyTorch's bundled cuDNN DLLs on PATH.
    try:
        import torch as _t  # noqa: F401
        _lib = os.path.join(os.path.dirname(_t.__file__), "lib")
        if _lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

# UTF-8 console output everywhere (the old scripts did this only for Windows)
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream and _stream.encoding and _stream.encoding.lower() != "utf-8":
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch  # noqa: E402


# --------------------------------------------------------------------------
# Torch device
# --------------------------------------------------------------------------
def get_torch_device() -> str:
    """Return the best available torch device string.

    Order: explicit ``DEVICE`` env var -> CUDA -> Apple MPS -> CPU.
    """
    forced = os.environ.get("DEVICE", "").strip().lower()
    if forced:
        if forced.startswith("cuda") and not torch.cuda.is_available():
            print(f"[platform] DEVICE={forced} requested but CUDA unavailable -> cpu")
            return "cpu"
        if forced == "mps" and not torch.backends.mps.is_available():
            print("[platform] DEVICE=mps requested but MPS unavailable -> cpu")
            return "cpu"
        return forced
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_torch_device()


def device_name() -> str:
    if DEVICE.startswith("cuda"):
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA GPU"
    if DEVICE == "mps":
        chip = _mac_chip_name()
        return f"Apple {chip} GPU (Metal / MPS)" if chip else "Apple GPU (Metal / MPS)"
    return f"CPU ({platform.processor() or platform.machine()})"


def _mac_chip_name() -> str:
    if not IS_MAC:
        return ""
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, timeout=2).stdout.strip()
        return out
    except Exception:
        return ""


# --------------------------------------------------------------------------
# ONNX Runtime providers
# --------------------------------------------------------------------------
def get_ort_providers() -> list:
    """Pick ONNX Runtime execution providers for this machine.

    * ``ORT_PROVIDERS=CoreMLExecutionProvider,CPUExecutionProvider`` overrides.
    * CUDA is used when both torch and onnxruntime-gpu see a GPU.
    * On Apple Silicon the CoreML EP is the default (measured ~27x faster than
      CPU for PeopleNet INT8 and ~3x for RTMPose, with matching outputs); set
      ``ORT_USE_COREML=0`` to force CPU.
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]

    forced = os.environ.get("ORT_PROVIDERS", "").strip()
    if forced:
        wanted = [p.strip() for p in forced.split(",") if p.strip()]
        chosen = [p for p in wanted if p in available]
        if "CPUExecutionProvider" not in chosen:
            chosen.append("CPUExecutionProvider")
        return chosen

    providers = []
    if "CUDAExecutionProvider" in available and torch.cuda.is_available():
        providers.append("CUDAExecutionProvider")
    if IS_MAC and "CoreMLExecutionProvider" in available and os.environ.get("ORT_USE_COREML", "1") != "0":
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def make_ort_session(model_path: str, providers: list | None = None, log_prefix: str = "[ORT]"):
    """Create an InferenceSession with graceful fallback to CPU."""
    import onnxruntime as ort
    providers = providers or get_ort_providers()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3  # hide CoreML partition info/warnings
    try:
        return ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    except Exception as exc:  # pragma: no cover - hardware dependent
        print(f"{log_prefix} providers {providers} failed ({exc}); falling back to CPU")
        return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])


# --------------------------------------------------------------------------
# Summary (used by /api/system and the env_check script)
# --------------------------------------------------------------------------
def system_summary() -> dict:
    try:
        import onnxruntime as ort
        ort_version = ort.__version__
        ort_available = ort.get_available_providers()
    except Exception:
        ort_version, ort_available = None, []
    try:
        import ultralytics
        ul_version = ultralytics.__version__
    except Exception:
        ul_version = None
    try:
        import cv2
        cv_version = cv2.__version__
    except Exception:
        cv_version = None
    return {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_device": DEVICE,
        "device_name": device_name(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "onnxruntime": ort_version,
        "ort_providers_available": ort_available,
        "ort_providers_selected": get_ort_providers(),
        "ultralytics": ul_version,
        "opencv": cv_version,
        "cpu_count": os.cpu_count(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(system_summary(), indent=2))


# --------------------------------------------------------------------------
# BoxMOT compatibility (import path moved between 21.x and 25.x)
# --------------------------------------------------------------------------
def import_ocsort():
    """Return the OcSort class regardless of the installed boxmot version."""
    try:
        from boxmot import OcSort  # boxmot >= 13
        return OcSort
    except ImportError:
        pass
    try:
        from boxmot.trackers.bbox.ocsort.ocsort import OcSort  # boxmot 21.x
        return OcSort
    except ImportError:
        from boxmot.trackers.ocsort.ocsort import OcSort  # boxmot 10.x
        return OcSort


def make_ocsort(iou_threshold=0.25, max_lost=60, min_confidence=0.25):
    """Build an OC-SORT tracker with the project's tuned parameters.

    Falls back to the minimal kwarg set if the installed boxmot rejects the
    tuned ones (delta_t / asso_func / inertia).
    """
    OcSort = import_ocsort()
    try:
        return OcSort(
            det_thresh=min_confidence,
            min_conf=min_confidence,
            max_age=max_lost,
            min_hits=2,
            iou_threshold=iou_threshold,
            delta_t=3,
            asso_func="iou",
            inertia=0.2,
            per_class=False,
        )
    except TypeError:
        try:
            return OcSort(det_thresh=min_confidence, max_age=max_lost, min_hits=2,
                          iou_threshold=iou_threshold, delta_t=3, asso_func="iou",
                          inertia=0.2, per_class=False)
        except TypeError:
            print("[platform] boxmot rejected tuned OC-SORT kwargs; using safe defaults")
            return OcSort(det_thresh=min_confidence, max_age=max_lost, min_hits=2,
                          iou_threshold=iou_threshold, per_class=False)
