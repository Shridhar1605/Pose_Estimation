#!/usr/bin/env bash
# One-shot launcher for macOS / Linux: backend (FastAPI, :8000) + dashboard (Vite, :5173)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "[run] creating .venv (python3.12) and installing requirements..."
  PY=$(command -v python3.12 || command -v python3)
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
[ -f _/resnet34_peoplenet_int8.onnx ] && [ -f yolo26n.pt ] || .venv/bin/python download_models.py

MODE="${1:-dev}"   # dev = vite dev server with HMR | prod = build once, serve from FastAPI
if [ "$MODE" = "prod" ]; then
  (cd dashboard && [ -d node_modules ] || npm install) 
  (cd dashboard && npm run build)
  echo "[run] dashboard built -> http://localhost:8000"
  exec .venv/bin/python server.py
fi

(cd dashboard && { [ -d node_modules ] || npm install; })
.venv/bin/python server.py &
BACK=$!
trap 'kill $BACK 2>/dev/null || true' EXIT INT TERM
(cd dashboard && npm run dev -- --host)
