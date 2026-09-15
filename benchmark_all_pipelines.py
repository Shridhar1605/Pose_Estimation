"""
benchmark_all_pipelines.py
===========================
Automated benchmark runner for CV pipelines using Pose_Samples videos.
Executes both Single Stream and Multi Stream (4 streams) for YOLO26n and PeopleNet.

For each pipeline & mode:
  1. Starts the pipeline on the backend via REST API
  2. Interacts with the frontend via Playwright to switch mode (Single / Multi 2x2)
  3. Captures frontend screenshots at 10-second intervals showing live analysis
  4. Collects benchmark stats (FPS per stream, total system throughput, person counts)
  5. Captures raw annotated frames from WebSocket stream
  6. Saves everything to pipeline_screenshots/<model_name>/<single_stream|multi_stream_4x>/
  7. Stops after max 60 seconds per run

Prerequisites:
  - Backend server (server.py) running on localhost:8000
  - Frontend dev server (dashboard) running on localhost:5173
  - Playwright chromium browser installed
"""

import os
import sys
import json
import time
import base64
import asyncio
import signal
import subprocess
import threading
import datetime
import traceback
from pathlib import Path

# Force stdout & stderr to use UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"
WS_URL = "ws://localhost:8000/ws/stream"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pipeline_screenshots")

# Target Models
TARGET_PIPELINES = ["peoplenet", "yolo26n"]

# Pose_Samples videos
SINGLE_VIDEO = "Pose_Samples/fight_21.mkv"
MULTI_VIDEOS = [
    "Pose_Samples/fight_21.mkv",
    "Pose_Samples/fight_22.mkv",
    "Pose_Samples/fall_19.mkv",
    "Pose_Samples/fall_20.mkv",
]

# Timing
PIPELINE_DURATION_S = 60      # max runtime per pipeline run (1 minute)
WARMUP_S = 6                  # seconds to wait after starting for model warm-up
SCREENSHOT_INTERVAL_S = 10    # take screenshot every N seconds
FRAME_CAPTURE_INTERVAL_S = 10 # capture raw WebSocket frame every N seconds

# ---------------------------------------------------------------------------
# Server / Frontend process management
# ---------------------------------------------------------------------------
server_proc = None
frontend_proc = None


def start_server():
    """Start the FastAPI backend server."""
    global server_proc
    print("\n[Benchmark] Starting backend server...")
    python_exe = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")          # macOS / Linux
    if not os.path.exists(python_exe):
        python_exe = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")  # Windows
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    server_proc = subprocess.Popen(
        [python_exe, "server.py"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        start_new_session=(os.name != "nt"),
    )
    # Wait for server to be ready
    for attempt in range(30):
        time.sleep(2)
        try:
            r = requests.get(f"{API_BASE}/api/models", timeout=5)
            if r.status_code == 200:
                print(f"[Benchmark] Server ready after {(attempt+1)*2}s")
                return True
        except requests.ConnectionError:
            pass
        print(f"[Benchmark] Waiting for server... ({(attempt+1)*2}s)")
    print("[Benchmark] ERROR: Server did not start in time!")
    return False


def start_frontend():
    """Start the Vite dev server for the dashboard."""
    global frontend_proc
    print("[Benchmark] Starting frontend dev server...")
    npm = "npm.cmd" if os.name == "nt" else "npm"
    frontend_proc = subprocess.Popen(
        [npm, "run", "dev"],
        cwd=os.path.join(PROJECT_ROOT, "dashboard"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        start_new_session=(os.name != "nt"),
    )
    # Wait for Vite to be ready
    for attempt in range(15):
        time.sleep(2)
        try:
            r = requests.get(FRONTEND_URL, timeout=5)
            if r.status_code == 200:
                print(f"[Benchmark] Frontend ready after {(attempt+1)*2}s")
                return True
        except requests.ConnectionError:
            pass
        print(f"[Benchmark] Waiting for frontend... ({(attempt+1)*2}s)")
    print("[Benchmark] WARNING: Frontend may not be ready. Continuing anyway...")
    return False


def stop_processes():
    """Stop both server and frontend processes."""
    global server_proc, frontend_proc
    print("\n[Benchmark] Stopping processes...")

    for name, proc in [("Frontend", frontend_proc), ("Server", server_proc)]:
        if proc and proc.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                    )
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
                print(f"  {name} stopped.")
            except Exception as e:
                print(f"  {name} stop error: {e}")
                try:
                    proc.kill()
                except:
                    pass

    server_proc = None
    frontend_proc = None


def restart_server_only():
    """Restart only the backend server while keeping the frontend running."""
    global server_proc
    if server_proc and server_proc.poll() is None:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(server_proc.pid)],
                    capture_output=True,
                )
            else:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            server_proc.wait(timeout=5)
            print("  [Server] Backend server reset.")
        except Exception as e:
            print(f"  [Server] Reset notice: {e}")
        server_proc = None
    time.sleep(1)
    start_server()


# ---------------------------------------------------------------------------
# WebSocket frame collector (runs in a background thread)
# ---------------------------------------------------------------------------
class MultiStreamFrameCollector:
    """Connects to WebSocket and collects frames & per-stream stats."""

    def __init__(self):
        self.frames = []          # list of (timestamp, stream_id, base64_jpeg)
        self.stats_log = []       # list of {stream_id, fps, person_count, ...}
        self.running = False
        self._thread = None
        self.total_ws_frames = 0
        self.stream_fps_samples = {}      # sid -> list of fps
        self.stream_person_samples = {}   # sid -> list of counts
        self.start_time = 0

    def start(self):
        self.frames = []
        self.stats_log = []
        self.total_ws_frames = 0
        self.stream_fps_samples = {}
        self.stream_person_samples = {}
        self.start_time = time.time()
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self):
        try:
            asyncio.run(self._ws_loop())
        except Exception as e:
            print(f"  [FrameCollector] Error: {e}")

    async def _ws_loop(self):
        import websockets

        try:
            async with websockets.connect(WS_URL, max_size=15 * 1024 * 1024) as ws:
                last_capture = {}
                while self.running:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        data = json.loads(msg)

                        if data.get("type") == "frame":
                            self.total_ws_frames += 1
                            sid = data.get("stream_id", "stream_0")
                            fps = data.get("fps", 0)
                            person_count = data.get("person_count", 0)
                            frame_number = data.get("frame_number", 0)

                            if sid not in self.stream_fps_samples:
                                self.stream_fps_samples[sid] = []
                                self.stream_person_samples[sid] = []
                                last_capture[sid] = 0

                            self.stream_fps_samples[sid].append(fps)
                            self.stream_person_samples[sid].append(person_count)

                            self.stats_log.append({
                                "timestamp": time.time(),
                                "stream_id": sid,
                                "fps": fps,
                                "person_count": person_count,
                                "frame_number": frame_number,
                                "model": data.get("model", ""),
                            })

                            # Capture frame every FRAME_CAPTURE_INTERVAL_S per stream
                            now = time.time()
                            if now - last_capture[sid] >= FRAME_CAPTURE_INTERVAL_S:
                                self.frames.append((now, sid, data.get("frame", "")))
                                last_capture[sid] = now

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        if self.running:
                            print(f"  [FrameCollector] WS error: {e}")
                        break
        except Exception as e:
            print(f"  [FrameCollector] Connection error: {e}")


# ---------------------------------------------------------------------------
# Browser screenshot capture via Playwright
# ---------------------------------------------------------------------------
async def capture_browser_screenshots(mode, output_folder, duration_s, interval_s):
    """
    Capture screenshots of the frontend at regular intervals using Playwright.
    Interacts with the frontend UI to toggle stream mode (Single / Multi 2x2).
    """
    screenshots = []
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": 1920, "height": 1080})

            print(f"    [Browser] Opening frontend URL ({FRONTEND_URL})...")
            await page.goto(FRONTEND_URL, wait_until="domcontentloaded", timeout=20000)

            # If multi-stream mode, click the Multi button in the UI
            if mode == "multi":
                try:
                    await asyncio.sleep(1)
                    multi_btn = page.locator('button:has-text("Multi")')
                    if await multi_btn.count() > 0:
                        await multi_btn.click()
                        print("    [Browser] Clicked 'Multi (2x2)' mode toggle in UI")
                except Exception as e:
                    print(f"    [Browser] Note on mode toggle: {e}")

            print(f"    [Browser] Waiting for video streams to render...")
            await asyncio.sleep(5)

            start_time = time.time()
            screenshot_num = 0

            while time.time() - start_time < duration_s:
                elapsed = time.time() - start_time
                next_capture = (screenshot_num + 1) * interval_s

                if elapsed >= next_capture:
                    screenshot_num += 1
                    prefix = "frontend_single" if mode == "single" else "frontend_multi4"
                    ss_name = f"{prefix}_{screenshot_num * interval_s}s.png"
                    ss_path = os.path.join(output_folder, ss_name)
                    await page.screenshot(path=ss_path, full_page=False)
                    screenshots.append(ss_path)
                    print(f"    [Screenshot] Captured UI: {os.path.basename(ss_path)}")

                await asyncio.sleep(1)

            await browser.close()

    except ImportError:
        print("  [Screenshots] Playwright not available. Skipping browser screenshots.")
    except Exception as e:
        print(f"  [Screenshots] Error capturing browser screenshots: {e}")
        traceback.print_exc()

    return screenshots


# ---------------------------------------------------------------------------
# Single / Multi Run Runner
# ---------------------------------------------------------------------------
def run_benchmark_session(model_name, mode, videos, output_folder):
    """Run a 60s benchmark session for a given model, mode, and list of videos."""
    mode_label = "SINGLE-STREAM (1 Stream)" if mode == "single" else f"MULTI-STREAM ({len(videos)} Streams)"
    print(f"\n  {'-'*60}")
    print(f"  RUNNING: {model_name.upper()} | {mode_label}")
    print(f"  Videos: {videos}")
    print(f"  {'-'*60}")

    os.makedirs(output_folder, exist_ok=True)

    result = {
        "model_name": model_name,
        "mode": mode,
        "stream_count": len(videos),
        "videos": videos,
        "status": "unknown",
        "error": None,
        "per_stream_stats": {},
        "system_avg_fps": 0,
        "total_frames_processed": 0,
        "total_ws_frames_received": 0,
        "duration_s": 0,
        "screenshots": [],
    }

    # 1. Start pipeline via REST API
    print(f"  Starting backend pipeline ({mode} mode)...")
    try:
        r = requests.post(
            f"{API_BASE}/api/start",
            json={"model": model_name, "mode": mode, "videos": videos},
            timeout=120,
        )
        if r.status_code != 200:
            err = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"error": r.text}
            result["status"] = "failed_to_start"
            result["error"] = str(err)
            print(f"  [ERROR] Failed to start: {err}")
            return result
        print(f"  [OK] Backend pipeline started")
    except Exception as e:
        result["status"] = "failed_to_start"
        result["error"] = str(e)
        print(f"  [ERROR] Failed to start: {e}")
        return result

    # 2. Start WebSocket frame collector
    collector = MultiStreamFrameCollector()
    collector.start()

    # 3. Wait for warm-up
    print(f"  [Wait] Warm-up phase ({WARMUP_S}s)...")
    time.sleep(WARMUP_S)

    # 4. Run benchmark & capture Playwright UI screenshots
    print(f"  [Run] Executing for {PIPELINE_DURATION_S}s with frontend UI capture...")
    benchmark_start = time.time()

    screenshot_paths = []
    try:
        loop = asyncio.new_event_loop()
        screenshot_paths = loop.run_until_complete(
            capture_browser_screenshots(mode, output_folder, PIPELINE_DURATION_S, SCREENSHOT_INTERVAL_S)
        )
        loop.close()
    except Exception as e:
        print(f"  [Screenshots] Browser capture error: {e}")
        remaining = PIPELINE_DURATION_S - (time.time() - benchmark_start)
        if remaining > 0:
            time.sleep(remaining)

    benchmark_end = time.time()
    actual_duration = benchmark_end - benchmark_start

    # 5. Stop collector and stop pipeline
    collector.stop()
    print(f"  [Stop] Stopping pipeline...")
    try:
        requests.post(f"{API_BASE}/api/stop", timeout=10)
    except Exception as e:
        print(f"  Warning: Stop request error: {e}")

    time.sleep(2)

    # 6. Save raw annotated WebSocket frames
    frame_paths = []
    for i, (ts, sid, frame_b64) in enumerate(collector.frames):
        if frame_b64:
            try:
                frame_bytes = base64.b64decode(frame_b64)
                elapsed = int(ts - collector.start_time)
                fname = f"frame_{sid}_{elapsed}s.jpg"
                fpath = os.path.join(output_folder, fname)
                with open(fpath, "wb") as f:
                    f.write(frame_bytes)
                frame_paths.append(fpath)
                print(f"    [Frame] Saved streaming frame: {fname}")
            except Exception as e:
                print(f"    Warning: Failed to save frame: {e}")

    # 7. Compute stats per stream
    total_frames = 0
    stream_averages = []

    for sid, fps_list in collector.stream_fps_samples.items():
        person_list = collector.stream_person_samples.get(sid, [])
        warmup_skip = min(10, len(fps_list) // 4)
        steady_fps = fps_list[warmup_skip:] if len(fps_list) > warmup_skip else fps_list

        avg_fps = round(sum(steady_fps) / len(steady_fps), 2) if steady_fps else 0
        peak_fps = round(max(fps_list), 2) if fps_list else 0
        avg_ppl = round(sum(person_list) / len(person_list), 2) if person_list else 0
        peak_ppl = max(person_list) if person_list else 0

        stream_averages.append(avg_fps)

        result["per_stream_stats"][sid] = {
            "avg_fps": avg_fps,
            "peak_fps": peak_fps,
            "avg_person_count": avg_ppl,
            "peak_person_count": peak_ppl,
            "ws_frames": len(fps_list),
        }

    if collector.stats_log:
        last_frames = {}
        for entry in collector.stats_log:
            last_frames[entry["stream_id"]] = entry["frame_number"]
        total_frames = sum(last_frames.values())

    result["system_avg_fps"] = round(sum(stream_averages), 2) if stream_averages else 0
    result["total_frames_processed"] = total_frames
    result["total_ws_frames_received"] = collector.total_ws_frames
    result["duration_s"] = round(actual_duration, 1)
    result["screenshots"] = screenshot_paths + frame_paths
    result["status"] = "success" if collector.total_ws_frames > 0 else "no_frames"

    # 8. Write benchmark_stats.txt
    stats_path = os.path.join(output_folder, "benchmark_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"  BENCHMARK RESULTS: {model_name.upper()} ({mode_label})\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Configuration\n")
        f.write(f"{'-'*40}\n")
        f.write(f"  Model Name         : {model_name}\n")
        f.write(f"  Mode               : {mode}\n")
        f.write(f"  Stream Count       : {len(videos)}\n")
        f.write(f"  Tracker            : OC-SORT\n")
        f.write(f"  Pose Estimator     : RTMPose-S (ONNX)\n")
        f.write(f"  Test Videos        : {', '.join(videos)}\n")
        f.write(f"  Run Date           : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Overall Metrics\n")
        f.write(f"{'-'*40}\n")
        f.write(f"  Status                  : {result['status']}\n")
        f.write(f"  Duration                : {result['duration_s']}s\n")
        f.write(f"  Combined System FPS     : {result['system_avg_fps']}\n")
        f.write(f"  Total Frames Processed  : {result['total_frames_processed']}\n")
        f.write(f"  Total WS Frames Recv    : {result['total_ws_frames_received']}\n\n")
        f.write(f"Per-Stream Metrics\n")
        f.write(f"{'-'*40}\n")
        for sid, sdata in result["per_stream_stats"].items():
            f.write(f"  {sid:<12} -> Avg FPS: {sdata['avg_fps']:<5.1f} | Peak FPS: {sdata['peak_fps']:<5.1f} | "
                    f"Avg Persons: {sdata['avg_person_count']:<5.1f} | Peak Persons: {sdata['peak_person_count']}\n")

        f.write(f"\nCaptured Files\n")
        f.write(f"{'-'*40}\n")
        for sp in screenshot_paths:
            f.write(f"  [Frontend UI Screenshot]  {os.path.basename(sp)}\n")
        for fp in frame_paths:
            f.write(f"  [WebSocket Frame]        {os.path.basename(fp)}\n")
        f.write(f"\n{'='*60}\n")

    print(f"  [Stats] Saved benchmark_stats.txt in: {output_folder}")
    return result


# ---------------------------------------------------------------------------
# Master Benchmark Orchestrator
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'#'*70}")
    print(f"  POSE_SAMPLES CV PIPELINE BENCHMARK (Single Stream & 4x Multi Stream)")
    print(f"  Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    # Scrub previous output
    print("[Benchmark] Scrubbing previous screenshots folder...")
    if os.path.exists(OUTPUT_DIR):
        try:
            import shutil
            shutil.rmtree(OUTPUT_DIR)
        except Exception as e:
            print(f"Warning clearing directory: {e}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check server / frontend
    server_was_running = False
    try:
        r = requests.get(f"{API_BASE}/api/models", timeout=3)
        if r.status_code == 200:
            server_was_running = True
            print("[Benchmark] Backend server is already running!")
    except requests.ConnectionError:
        pass

    frontend_was_running = False
    try:
        r = requests.get(FRONTEND_URL, timeout=3)
        if r.status_code == 200:
            frontend_was_running = True
            print("[Benchmark] Frontend is already running!")
    except requests.ConnectionError:
        pass

    if not server_was_running:
        if not start_server():
            print("[Benchmark] FATAL: Server failed to start.")
            stop_processes()
            return

    if not frontend_was_running:
        start_frontend()

    try:
        all_results = []

        for model_name in TARGET_PIPELINES:
            print(f"\n{'='*70}")
            print(f"  BENCHMARKING MODEL: {model_name.upper()}")
            print(f"{'='*70}")

            # 1. Single Stream Run
            restart_server_only()
            single_folder = os.path.join(OUTPUT_DIR, model_name, "single_stream")
            res_single = run_benchmark_session(model_name, "single", [SINGLE_VIDEO], single_folder)
            all_results.append(res_single)

            time.sleep(2)

            # 2. Multi Stream (4x) Run
            restart_server_only()
            multi_folder = os.path.join(OUTPUT_DIR, model_name, "multi_stream_4x")
            res_multi = run_benchmark_session(model_name, "multi", MULTI_VIDEOS, multi_folder)
            all_results.append(res_multi)

            time.sleep(2)

        # Write overall summary
        summary_path = os.path.join(OUTPUT_DIR, "SUMMARY.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"{'='*80}\n")
            f.write(f"  POSE_SAMPLES CV PIPELINE BENCHMARK SUMMARY\n")
            f.write(f"  Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"{'Model':<12} {'Mode':<15} {'Streams':<8} {'Status':<10} {'Combined FPS':>14} {'Frames':>10}\n")
            f.write(f"{'-'*12} {'-'*15} {'-'*8} {'-'*10} {'-'*14} {'-'*10}\n")

            for r in all_results:
                f.write(f"{r['model_name']:<12} {r['mode']:<15} {r['stream_count']:<8} "
                        f"{r['status']:<10} {r['system_avg_fps']:>14.1f} {r['total_frames_processed']:>10}\n")

        print(f"\n[Summary] Overall summary written to: {summary_path}")

        print(f"\n\n{'#'*70}")
        print(f"  BENCHMARK COMPLETE!")
        print(f"  Results saved in: {OUTPUT_DIR}")
        print(f"{'#'*70}\n")

    finally:
        if not server_was_running or not frontend_was_running:
            stop_processes()


if __name__ == "__main__":
    main()
