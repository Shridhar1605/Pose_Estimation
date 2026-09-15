#!/usr/bin/env python3
"""
download_models.py - fetch every model the unified pipeline needs.

  yolo26n.pt                       Ultralytics YOLO26-nano (GitHub release asset)
  _/resnet34_peoplenet_int8.onnx   NVIDIA PeopleNet ResNet34 INT8 (public NGC asset)
  _/labels.txt, _/nvinfer_config.txt   PeopleNet metadata
  rtmpose-s.onnx                   RTMPose-s (SimCC, 256x192) keypoint model - optional,
                                   copied from a local path if given (--rtmpose /path/file.onnx)

Usage:
  .venv/bin/python download_models.py            # yolo26n + peoplenet
  .venv/bin/python download_models.py --yolo yolo26s yolo26m
"""
import argparse
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
NGC = "https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_decrypted_v2.3.4/files"
YOLO_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


def fetch(url: str, dest: str, min_bytes: int = 1) -> bool:
    if os.path.isfile(dest) and os.path.getsize(dest) >= min_bytes:
        print(f"  [skip] {os.path.relpath(dest, HERE)} already present")
        return True
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"  [get ] {url}")
    try:
        with urllib.request.urlopen(url, timeout=600) as r, open(dest + ".part", "wb") as f:
            shutil.copyfileobj(r, f)
        if os.path.getsize(dest + ".part") < min_bytes:
            raise IOError("downloaded file too small")
        os.replace(dest + ".part", dest)
        print(f"  [ ok ] {os.path.relpath(dest, HERE)} ({os.path.getsize(dest) / 1e6:.1f} MB)")
        return True
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        for p in (dest + ".part",):
            if os.path.exists(p):
                os.remove(p)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", nargs="*", default=["yolo26n"], help="YOLO26 variants to fetch (yolo26n/s/m/l/x)")
    ap.add_argument("--rtmpose", help="local path to rtmpose-s.onnx to copy into the project")
    args = ap.parse_args()

    ok = True
    print("PeopleNet (NVIDIA NGC, public):")
    ok &= fetch(f"{NGC}/resnet34_peoplenet_int8.onnx", os.path.join(HERE, "_", "resnet34_peoplenet_int8.onnx"), 1_000_000)
    for meta in ("labels.txt", "nvinfer_config.txt"):
        fetch(f"{NGC}/{meta}", os.path.join(HERE, "_", meta))

    print("YOLO26 (Ultralytics):")
    for name in args.yolo:
        ok &= fetch(f"{YOLO_RELEASE}/{name}.pt", os.path.join(HERE, f"{name}.pt"), 1_000_000)

    rtm = os.path.join(HERE, "rtmpose-s.onnx")
    if args.rtmpose and os.path.isfile(args.rtmpose):
        shutil.copy(args.rtmpose, rtm)
        print(f"RTMPose: copied {args.rtmpose}")
    elif os.path.isfile(rtm):
        print("RTMPose: rtmpose-s.onnx present")
    else:
        print("RTMPose: rtmpose-s.onnx NOT found - export it from MMPose/rtmlib (SimCC 256x192, inputs 'input', "
              "outputs 'simcc_x','simcc_y') and place it in the project root. The pipeline runs without it "
              "(action classification falls back to placeholder keypoints).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
