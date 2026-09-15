# Running on Windows

This guide sets up the unified pipeline (YOLO26 + NVIDIA PeopleNet, OC-SORT, RTMPose, action rules) and the live dashboard on **Windows 10 or 11, 64-bit**.
For macOS and Linux, see [README.md](README.md).

> **Status:** the code is Windows-compatible by design. The project was originally written on Windows,
> every dependency resolves to a Windows wheel for Python 3.12, and all OS-specific code paths have Windows branches.
> The macOS build is the one that has been run end to end; if you hit a Windows-only problem, check [Troubleshooting](#troubleshooting).

## What runs where on Windows

| Component | NVIDIA GPU | AMD / Intel GPU | CPU only |
|---|---|---|---|
| YOLO26 (PyTorch) | CUDA | CPU | CPU |
| PeopleNet, RTMPose (ONNX Runtime) | CUDA | DirectML (optional) | CPU |

Device selection is automatic ([platform_utils.py](platform_utils.py)): CUDA if available, otherwise CPU.
Without an NVIDIA GPU, **PeopleNet with DirectML** is usually the faster detector, since YOLO falls back to CPU.

## 1. Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | **3.12** (64-bit) | From [python.org](https://www.python.org/downloads/windows/). Tick *Add python.exe to PATH* and keep the *py launcher*. |
| Git | any recent | [git-scm.com](https://git-scm.com/download/win) |
| Node.js | 22 LTS or newer | [nodejs.org](https://nodejs.org/). Only needed for the dashboard. |
| NVIDIA driver | 570 or newer | Only for CUDA. No separate CUDA Toolkit install is needed; PyTorch ships the runtime. |

Git LFS is **not** required. The largest model is 44 MB.

## 2. Get the code

Open **PowerShell** and run:

```powershell
git clone https://github.com/Shridhar1605/Pose_Estimation.git
cd Pose_Estimation
```

All models are included in the repository (`*.pt` in the root, PeopleNet in `_\`, `rtmpose-s.onnx`). Videos are not.

## 3. Create the virtual environment

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

The commands below call `.venv\Scripts\python.exe` directly, so you never need to activate the venv.
If you prefer activating it and PowerShell blocks the script, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.venv\Scripts\Activate.ps1
```

## 4. Install dependencies

Pick **one** option. Order matters for the NVIDIA option: install CUDA PyTorch *before* the requirements file,
otherwise pip keeps the CPU-only build from PyPI.

### Option A: NVIDIA GPU (CUDA)

```powershell
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip uninstall -y onnxruntime
.venv\Scripts\python.exe -m pip install onnxruntime-gpu
```

### Option B: AMD or Intel GPU (DirectML for the ONNX models)

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip uninstall -y onnxruntime
.venv\Scripts\python.exe -m pip install onnxruntime-directml
```

DirectML is not picked automatically. Enable it for the session before starting the server:

```powershell
$env:ORT_PROVIDERS = "DmlExecutionProvider,CPUExecutionProvider"
```

### Option C: CPU only

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Verify

```powershell
.venv\Scripts\python.exe env_check.py
```

Check `torch_device` (`cuda:0` for Option A, `cpu` otherwise) and `ort_providers_selected`
(`CUDAExecutionProvider`, `DmlExecutionProvider`, or `CPUExecutionProvider` first).

## 5. Add videos

Put demo footage in either folder; both are scanned automatically and ignored by git:

```
Pose_Samples\     fall and fight clips
Video_samples\    any other footage
```

```powershell
mkdir Pose_Samples, Video_samples -Force
```

Supported formats: `.mp4 .mkv .avi .mov .webm .m4v .mpeg .mpg .flv`.
You can also upload a file from the dashboard, add a folder by absolute path such as `D:\footage`, or use an RTSP URL or `webcam:0`.

If a model file is ever missing, restore PeopleNet and YOLO26n with:

```powershell
.venv\Scripts\python.exe download_models.py
```

## 6. Run

`run.sh` is bash-only. On Windows, use two PowerShell windows.

**Window 1: backend**

```powershell
cd Pose_Estimation
.venv\Scripts\python.exe server.py
```

Wait for `Uvicorn running on http://0.0.0.0:8000`.

**Window 2: dashboard**

```powershell
cd Pose_Estimation\dashboard
npm install
npm run dev
```

Open **http://localhost:5173**, pick a detector, choose *Single feed* or *Quad 2×2*, tick your sources, and press **Start pipeline**.

### Single-window alternative

Build the dashboard once and let the backend serve it on one port:

```powershell
cd Pose_Estimation\dashboard
npm install
npm run build
cd ..
.venv\Scripts\python.exe server.py
```

Then open **http://localhost:8000**. Re-run `npm run build` after changing dashboard code.

## Environment variables (PowerShell syntax)

Set these in the same window before `server.py`. They last for that window only.

| Command | Effect |
|---|---|
| `$env:DEVICE = "cpu"` | Force PyTorch to CPU (or `"cuda:0"`). |
| `$env:ORT_PROVIDERS = "CUDAExecutionProvider,CPUExecutionProvider"` | Explicit ONNX Runtime provider order. |
| `$env:PORT = "8010"` | Serve on a different port. |
| `$env:HOST = "127.0.0.1"` | Listen on this PC only. Avoids the firewall prompt. |

## Troubleshooting

**Port 8000 already in use.** Another server is still running. Find and stop it:

```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

Or start on another port with `$env:PORT = "8010"`.

**Windows Defender Firewall prompt on first start.** The server listens on all interfaces so other devices can view the dashboard.
Allow *Private networks*, or set `$env:HOST = "127.0.0.1"` to keep it local.

**`torch.cuda.is_available()` is `False` on an NVIDIA machine.** The CPU build of PyTorch got installed.
Reinstall the CUDA build, then confirm:

```powershell
.venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**ONNX models run on CPU despite `onnxruntime-gpu`.** The server logs the providers each model actually loaded with, for example `[PeopleNet] loaded ... | providers: [...]`.
If CUDA failed to load, it falls back to CPU instead of crashing. Common causes:
both `onnxruntime` and `onnxruntime-gpu` installed at once (uninstall both, then reinstall only `onnxruntime-gpu`),
or an NVIDIA driver older than 570.

**`py -3.12` is not recognized.** Python 3.12 was installed without the py launcher.
Use the full path instead, for example `& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv`.

**Webcam source shows "Cannot open source".** Allow camera access under
*Settings → Privacy & security → Camera → Let desktop apps access your camera*, and close other apps using the camera.

**`.mkv` or `.mov` files won't open.** The `opencv-python` wheel bundles FFmpeg on Windows, so this is usually a corrupt or partially downloaded file.
Run `.venv\Scripts\python.exe env_check.py` to confirm OpenCV is installed, then try the file in VLC.

**Benchmark runner.** `benchmark_all_pipelines.py` has Windows process handling built in. It needs a browser for Playwright:

```powershell
.venv\Scripts\python.exe -m pip install playwright
.venv\Scripts\python.exe -m playwright install chromium
```
