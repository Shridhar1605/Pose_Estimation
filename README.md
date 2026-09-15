# Neelaminds Vision — unified person detection · tracking · action pipeline

One pipeline, two interchangeable detectors, one live dashboard.

```
 video / RTSP / webcam ─▶ detector ─▶ OC-SORT ─▶ RTMPose ─▶ action rules ─▶ JPEG + JSON over WebSocket ─▶ dashboard
                          ├ YOLO26 (.pt, PyTorch)                (STANDING / SITTING / LYING DOWN / FIGHTING)
                          └ NVIDIA PeopleNet (ONNX INT8)
```

The code base originally targeted Windows/Linux with CUDA. It now runs natively on
macOS (Apple Silicon and Intel) as well as CUDA machines and plain CPU:

| Component | macOS (Apple Silicon) | Linux / Windows + NVIDIA | CPU only |
|---|---|---|---|
| YOLO26 (PyTorch) | Metal / MPS | CUDA | CPU |
| PeopleNet, RTMPose (ONNX Runtime) | CoreML EP (auto) | CUDA EP (`onnxruntime-gpu`) | CPU EP |

Device selection lives in one place, [platform_utils.py](platform_utils.py), and every script imports it.

## Quick start (macOS / Linux)

```bash
cd Pose-Identinfication
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_models.py            # PeopleNet from NVIDIA NGC + yolo26n.pt from Ultralytics
python server.py                     # API on http://localhost:8000
```

Dashboard (in a second terminal):

```bash
cd dashboard
npm install
npm run dev                          # http://localhost:5173  (proxies /api and /ws to :8000)
# or: npm run build   -> server.py then serves the built app at http://localhost:8000
```

Or simply `./run.sh` (dev, hot reload) / `./run.sh prod` (build once, single port 8000).

Windows: `py -3.12 -m venv .venv && .venv\Scripts\activate`, then the same `pip`/`python` commands.
For NVIDIA GPUs see [requirements-cuda.txt](requirements-cuda.txt).

## Models

| File | What | Source |
|---|---|---|
| `_/resnet34_peoplenet_int8.onnx` | NVIDIA PeopleNet ResNet34, INT8, 960×544, classes person/bag/face | NGC `nvidia/tao/peoplenet` (`download_models.py`) |
| `yolo26n.pt` | Ultralytics YOLO26-nano, COCO | GitHub release (`download_models.py`) |
| `yolo26m_person.pt` | YOLO26-m fine-tuned single-class person (CCTV) | project checkpoint |
| `yolo26_finetuned_best.pt` | YOLO26 fine-tuned `lying_person` / `sitting_person` | project checkpoint |
| `rtmpose-s.onnx` | RTMPose-s SimCC 256×192, 17 COCO keypoints | project asset (optional; pipeline falls back to placeholder keypoints) |

Every `*.pt` in the project root is discovered automatically and offered in the dashboard.

## Measured on an Apple M3 (8-core, 720p sources)

| Stage | Backend | Latency |
|---|---|---|
| PeopleNet INT8 | ONNX Runtime · CoreML | ~6.5 ms (CPU EP: ~176 ms) |
| YOLO26n | PyTorch · MPS | ~12 ms (CPU: ~42 ms) |
| YOLO26m person | PyTorch · MPS | ~37 ms |
| RTMPose-s per crop | ONNX Runtime · CoreML | ~4 ms |

Four simultaneous streams run at the sources' native frame rate with either detector.

## Environment overrides

| Variable | Effect |
|---|---|
| `DEVICE=cpu` / `mps` / `cuda:0` | force the torch device |
| `ORT_USE_COREML=0` | disable the CoreML execution provider on macOS |
| `ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider` | explicit ONNX Runtime provider list |
| `PORT=8000`, `HOST=0.0.0.0` | server bind address |

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/system` | hardware, versions, selected providers, models |
| GET | `/api/models` | detectors available |
| GET | `/api/videos` | discovered sources (`Video_samples/`, `videos/`, `Pose_Samples/`, added folders) |
| POST | `/api/folders` | `{ "path": "/abs/folder" }` add a folder to scan |
| POST | `/api/upload` | multipart video upload |
| POST | `/api/start` | `{ "model": "peoplenet", "mode": "single" \| "multi", "videos": ["Video_samples/x.mp4", "rtsp://…", "webcam:0"] }` |
| POST | `/api/stop` | stop all streams |
| GET | `/api/status` | per-stream FPS, latency, counts |
| WS | `/ws/stream` | `{type:"frame", stream_id, frame (base64 JPEG), fps, latency_ms, person_count, unique_persons, actions, tracks[]}` |

Interactive docs: http://localhost:8000/docs

## Project layout

| Path | Role |
|---|---|
| `server.py` | **the unified pipeline + API** (both detectors, tracking, pose, actions, streaming) |
| `platform_utils.py` | cross-platform device / ONNX Runtime / boxmot compatibility |
| `download_models.py` | fetches PeopleNet + YOLO26 weights |
| `dashboard/` | React + Vite operator console |
| `pipelinePrototype.py`, `PeopleNetProto.py` | original standalone batch scripts (YOLO / PeopleNet), now cross-platform |
| `PeopleNetDet.py`, `PersonOCSort.py`, `PersonByteTrack.py`, `basicPersonDet.py`, `personDetection.py` | earlier experiments, kept runnable |
| `PeopleNetBenchmark.py`, `benchmark_all_pipelines.py` | benchmarks (the latter drives the dashboard with Playwright) |
| `env_check.py`, `cuda_check.py`, `gpu_check.py` | environment diagnostics |
| `Video_samples/` | demo footage (OpenCV `vtest.avi`, Intel IoT sample videos) — git-ignored |

## Notes on the macOS port

- `torch.cuda` checks were replaced by `platform_utils.get_torch_device()` (CUDA → MPS → CPU).
- ONNX sessions are created through `make_ort_session()` which picks CUDA / CoreML / CPU and falls back safely.
- `boxmot` moved OC-SORT between releases; `make_ocsort()` handles 21.x and 25.x.
- Windows-only DLL path hacks, `cp1252` console fixes, `.venv\Scripts` paths, `shell=True` npm spawning
  and `F:\...` absolute paths were removed or made conditional.
- `KMP_DUPLICATE_LIB_OK` is set to avoid the duplicate-libomp abort when torch, onnxruntime and OpenCV coexist.
- Webcam capture uses AVFoundation on macOS (`webcam:0` as a source; grant camera permission to the terminal app).
