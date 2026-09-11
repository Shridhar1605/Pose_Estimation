"""
PeopleNetBenchmark.py
=====================
Comprehensive benchmarking tool for the PeopleNet detection pipeline.

Modes
-----
* **Single-stream** : Full pipeline (PeopleNet - OC-SORT - RTMPose - Action
  Classification) benchmark on individual videos.
* **Multi-stream**  : Parallel processing via ThreadPoolExecutor - automatically
  discovers the maximum number of concurrent camera streams that keep the
  total per-cycle delay below a 5-second threshold.

Usage
-----
    python PeopleNetBenchmark.py

Base: PeopleNetProto.py (imported, never modified)
"""

import os
import sys
import time
import random
import datetime
import threading
import platform
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import the pipeline components from PeopleNetProto.py
# The import triggers model loading (peoplenet + rtmpose singletons) and a
# "Using device: ..." print - that's expected.
# ---------------------------------------------------------------------------
from PeopleNetProto import (
    PeopleNetDetector,
    RTMPoseWrapper,
    OCSortTracker,
    classify_action,
    crop_with_padding,
    detect_persons_robust,
    peoplenet,
    rtmpose,
    draw_skeleton,
)

# -
# Constants
# -
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".ts", ".flv"}
WARMUP_FRAMES    = 2          # first N frames excluded from stats (JIT / cache warm-up)
BENCHMARK_CAP_S  = 5.0        # max wall-clock seconds per benchmark run
MULTI_THRESHOLD  = 5.0        # total per-cycle delay threshold in seconds
PROBE_FRAMES     = 10         # frames per stream during quick probe
INFER_EVERY_N    = 2          # run detection + pose every N-th frame (matches PeopleNetProto)

ACTION_COLORS = {
    "STANDING":  (0, 255, 0),    # Green
    "SITTING":   (0, 255, 255),  # Yellow
    "LYING DOWN": (0, 0, 255),   # Red
    "FIGHTING":  (0, 128, 255),  # Orange
    "UNKNOWN":   (128, 128, 128),
}

def draw_annotations(frame, tracked, track_states, track_history):
    """Draw bounding boxes, actions, and skeletons on a frame."""
    marked = frame.copy()
    for track in tracked:
        x1, y1, x2, y2 = map(int, track["bbox"])
        track_id = track["id"]
        state_info = track_states.get(track_id, {"action": "STANDING", "conf": 0.5})
        action = state_info["action"]
        action_conf = state_info["conf"]

        color = ACTION_COLORS.get(action, (0, 255, 0))
        label = f"ID:{track_id} {action} ({action_conf:.2f})"
        cv2.rectangle(marked, (x1, y1), (x2, y2), color, 2)
        cv2.putText(marked, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if len(track_history.get(track_id, [])) > 0:
            last_kps = track_history[track_id][-1].get("kps")
            last_rx1 = track_history[track_id][-1].get("rx1", 0)
            last_ry1 = track_history[track_id][-1].get("ry1", 0)
            draw_skeleton(marked, last_kps, last_rx1, last_ry1)
    
    count_label = f"Active Persons: {len(tracked)}"
    cv2.putText(marked, count_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return marked

def create_grid_image(frames_dict, num_streams, cell_size=(640, 360)):
    """Combine frames from multiple streams into a grid."""
    if not frames_dict:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)
    
    cols = int(np.ceil(np.sqrt(num_streams)))
    rows = int(np.ceil(num_streams / cols))
    
    grid_w = cols * cell_size[0]
    grid_h = rows * cell_size[1]
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for sid in range(1, num_streams + 1):
        if sid in frames_dict and frames_dict[sid] is not None:
            resized = cv2.resize(frames_dict[sid], cell_size)
            r = (sid - 1) // cols
            c = (sid - 1) % cols
            y = r * cell_size[1]
            x = c * cell_size[0]
            grid[y:y+cell_size[1], x:x+cell_size[0]] = resized
            
            # Put stream label
            cv2.putText(grid, f"Stream {sid}", (x + 10, y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
    return grid
# -
# Utility helpers
# -
def discover_videos(folder_path):
    """Recursively discover all video files in *folder_path*."""
    videos = []
    for root, _dirs, files in os.walk(folder_path):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in VIDEO_EXTENSIONS:
                videos.append(os.path.join(root, fname))
    return sorted(videos)


def _compute_percentile(sorted_values, percentile):
    """Return the *percentile*-th value from an already-sorted list."""
    if not sorted_values:
        return 0.0
    idx = min(int(len(sorted_values) * percentile / 100.0), len(sorted_values) - 1)
    return sorted_values[idx]


def _video_meta(video_path):
    """Return (width, height, fps, total_frames) for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, n


def _stats_from_latencies(latencies, wall_time):
    """Compute a stats dict from a list of per-frame latencies."""
    if not latencies:
        return {
            "frames": 0, "wall_time_s": wall_time,
            "avg_ms": 0, "min_ms": 0, "max_ms": 0,
            "median_ms": 0, "p95_ms": 0, "p99_ms": 0, "fps": 0,
        }
    s = sorted(latencies)
    avg = sum(s) / len(s)
    fps = len(s) / wall_time if wall_time > 0 else 0
    return {
        "frames":    len(s),
        "wall_time_s": wall_time,
        "avg_ms":    avg * 1000,
        "min_ms":    s[0] * 1000,
        "max_ms":    s[-1] * 1000,
        "median_ms": _compute_percentile(s, 50) * 1000,
        "p95_ms":    _compute_percentile(s, 95) * 1000,
        "p99_ms":    _compute_percentile(s, 99) * 1000,
        "fps":       fps,
    }


def _print_stats(title, stats):
    """Pretty-print a latency summary table to console."""
    print()
    print("=" * 66)
    print(f"  {title}")
    print("=" * 66)
    print(f"  {'Metric':<34} {'Value':>22}")
    print("-" * 66)
    print(f"  {'Frames processed':<34} {stats['frames']:>22}")
    print(f"  {'Total wall-clock time':<34} {stats['wall_time_s']:>21.3f}s")
    print(f"  {'Avg latency / frame':<34} {stats['avg_ms']:>20.2f}ms")
    print(f"  {'Min latency':<34} {stats['min_ms']:>20.2f}ms")
    print(f"  {'Max latency':<34} {stats['max_ms']:>20.2f}ms")
    print(f"  {'Median (P50)':<34} {stats['median_ms']:>20.2f}ms")
    print(f"  {'P95 latency':<34} {stats['p95_ms']:>20.2f}ms")
    print(f"  {'P99 latency':<34} {stats['p99_ms']:>20.2f}ms")
    print(f"  {'Throughput (FPS)':<34} {stats['fps']:>22.2f}")
    print("=" * 66)


# -
# Single-stream benchmark
# -
def benchmark_single_stream(video_path, max_frames=None, display=False):
    """
    Run the full detection pipeline on a single video and return benchmark
    statistics.

    Pipeline per frame (mirrors PeopleNetProto.process_video):
        1. PeopleNet detection  (every INFER_EVERY_N-th frame)
        2. OC-SORT tracking
        3. RTMPose pose estimation on each tracked person
        4. Action classification

    Returns
    -------
    dict with keys: video, stats (from _stats_from_latencies), latencies (raw).
    """
    w, h, vid_fps, total_frames = _video_meta(video_path)
    limit = max_frames or total_frames

    print(f"\n  Video       : {video_path}")
    print(f"  Resolution  : {w}x{h} @ {vid_fps:.1f} FPS")
    print(f"  Total frames: {total_frames}  (benchmarking up to {limit})")
    print()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    tracker       = OCSortTracker(iou_threshold=0.25, max_lost=60, min_confidence=0.25)
    track_history = {}
    track_states  = {}
    tracked       = []

    latencies  = []
    frame_idx  = 0
    wall_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop the video if we haven't hit the limit yet
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        if max_frames and frame_idx >= max_frames + WARMUP_FRAMES:
            break
        if time.perf_counter() - wall_start >= BENCHMARK_CAP_S:
            break

        t0 = time.perf_counter()

        # --- Full pipeline (matching PeopleNetProto.process_video) ---
        if frame_idx % INFER_EVERY_N == 0:
            dets    = detect_persons_robust(frame)
            tracked = tracker.update(dets.tolist() if len(dets) > 0 else [], frame)

        for track in tracked:
            tid  = track["id"]
            bbox = track["bbox"]

            if tid not in track_history:
                track_history[tid] = []
            if tid not in track_states:
                track_states[tid] = {"action": "STANDING", "conf": 0.5}

            if frame_idx % INFER_EVERY_N == 0:
                roi, rx1, ry1 = crop_with_padding(frame, bbox, pad=20)
                if roi.size > 0:
                    keypoints, scores = rtmpose.infer(roi)
                    bbox_height = bbox[3] - bbox[1]

                    track_history[tid].append({
                        "frame": frame_idx, "kps": keypoints,
                        "rx1": rx1, "ry1": ry1, "conf": scores,
                    })
                    if len(track_history[tid]) > 30:
                        track_history[tid].pop(0)

                    action, action_conf = classify_action(
                        keypoints, track_history[tid], bbox_height
                    )
                    track_states[tid] = {"action": action, "conf": action_conf}

        t1 = time.perf_counter()

        if display:
            marked = draw_annotations(frame, tracked, track_states, track_history)
            cv2.imshow("Single-Stream Benchmark", marked)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_idx >= WARMUP_FRAMES:
            latencies.append(t1 - t0)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"    [{os.path.basename(video_path)}] Processed {frame_idx} frames...")

    wall_end = time.perf_counter()
    cap.release()
    if display:
        cv2.destroyWindow("Single-Stream Benchmark")
        cv2.waitKey(1)

    wall_time = wall_end - wall_start
    stats = _stats_from_latencies(latencies, wall_time)

    _print_stats(f"Single-Stream - {os.path.basename(video_path)}", stats)

    return {"video": video_path, "stats": stats, "latencies": latencies}


# -
# Multi-stream worker
# -
def _stream_worker(stream_id, video_path, max_frames, results_dict, lock, display=False, display_frames=None, display_lock=None):
    """
    Independent worker for one camera stream.

    Each worker has its **own** VideoCapture, OCSortTracker, track history,
    and track state - fully independent pipeline.  Only the PeopleNet and
    RTMPose ONNX sessions (GPU singletons) are shared across threads.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [Stream {stream_id}] ERROR: Cannot open {video_path}")
        return

    tracker       = OCSortTracker(iou_threshold=0.25, max_lost=60, min_confidence=0.25)
    track_history = {}
    track_states  = {}
    tracked       = []

    latencies  = []
    frame_idx  = 0
    wall_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        if max_frames and frame_idx >= max_frames + WARMUP_FRAMES:
            break
        if time.perf_counter() - wall_start >= BENCHMARK_CAP_S:
            break

        t0 = time.perf_counter()

        if frame_idx % INFER_EVERY_N == 0:
            dets    = detect_persons_robust(frame)
            tracked = tracker.update(dets.tolist() if len(dets) > 0 else [], frame)

        for track in tracked:
            tid  = track["id"]
            bbox = track["bbox"]

            if tid not in track_history:
                track_history[tid] = []
            if tid not in track_states:
                track_states[tid] = {"action": "STANDING", "conf": 0.5}

            if frame_idx % INFER_EVERY_N == 0:
                roi, rx1, ry1 = crop_with_padding(frame, bbox, pad=20)
                if roi.size > 0:
                    keypoints, scores = rtmpose.infer(roi)
                    bbox_height = bbox[3] - bbox[1]

                    track_history[tid].append({
                        "frame": frame_idx, "kps": keypoints,
                        "rx1": rx1, "ry1": ry1, "conf": scores,
                    })
                    if len(track_history[tid]) > 30:
                        track_history[tid].pop(0)

                    action, action_conf = classify_action(
                        keypoints, track_history[tid], bbox_height
                    )
                    track_states[tid] = {"action": action, "conf": action_conf}

        t1 = time.perf_counter()
        
        if display and display_frames is not None and display_lock is not None:
            marked = draw_annotations(frame, tracked, track_states, track_history)
            with display_lock:
                display_frames[stream_id] = marked
                
        if frame_idx >= WARMUP_FRAMES:
            latencies.append(t1 - t0)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"    [Stream {stream_id}] Processed {frame_idx} frames...")

    cap.release()
    with lock:
        results_dict[stream_id] = {
            "video": video_path,
            "latencies": latencies,
        }


# -
# Multi-stream benchmark
# -
def benchmark_multi_stream(video_paths, num_streams, max_frames=None, display=False):
    """
    Launch *num_streams* parallel workers via ThreadPoolExecutor.

    Parameters
    ----------
    video_paths : list[str]
        Pool of videos to choose from.  Each stream picks a random video.
    num_streams : int
        Number of concurrent camera streams.
    max_frames : int or None
        Per-stream frame cap.
    display : bool
        Whether to show the composite multi-stream display.

    Returns
    -------
    dict with per-stream stats, aggregate stats, and the total delay metric.
    """
    # Assign a random video to each stream
    stream_videos = [random.choice(video_paths) for _ in range(num_streams)]

    print(f"\n{'-' * 66}")
    print(f"  Multi-Stream Benchmark - {num_streams} stream(s)")
    print(f"{'-' * 66}")
    for sid, vp in enumerate(stream_videos, 1):
        print(f"  Stream {sid}: {os.path.basename(vp)}")
    print()

    results_dict = {}
    lock = threading.Lock()
    
    display_frames = {} if display else None
    display_lock = threading.Lock() if display else None

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_streams) as pool:
        futures = []
        for sid in range(1, num_streams + 1):
            fut = pool.submit(
                _stream_worker,
                sid,
                stream_videos[sid - 1],
                max_frames,
                results_dict,
                lock,
                display,
                display_frames,
                display_lock
            )
            futures.append(fut)
            
        if display:
            while not all(f.done() for f in futures):
                with display_lock:
                    grid = create_grid_image(display_frames, num_streams)
                if grid is not None and grid.size > 0:
                    cv2.imshow("Multi-Stream Benchmark", grid)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            cv2.destroyWindow("Multi-Stream Benchmark")
            cv2.waitKey(1)
            
        for fut in as_completed(futures):
            fut.result()  # re-raise exceptions
    wall_end = time.perf_counter()
    total_wall = wall_end - wall_start
    
    gc.collect()

    # - Per-stream reports -
    per_stream_stats = {}
    all_latencies    = []
    for sid in sorted(results_dict.keys()):
        lats       = results_dict[sid]["latencies"]
        stream_wall = sum(lats) if lats else 0.0
        st         = _stats_from_latencies(lats, stream_wall)
        per_stream_stats[sid] = {
            "video": results_dict[sid]["video"],
            "stats": st,
        }
        _print_stats(f"Stream {sid} - {os.path.basename(results_dict[sid]['video'])}", st)
        all_latencies.extend(lats)

    # - Aggregate -
    agg_stats = _stats_from_latencies(all_latencies, total_wall)

    # Total delay = average per-frame latency across all streams combined.
    # When processing N streams in parallel threads sharing a single GPU,
    # the wall-clock time for one "cycle" - avg_latency (GPU is serialized).
    # We define total_delay = avg_latency_per_frame * num_streams  
    # representing worst-case round-robin servicing time.
    avg_lat_s    = (sum(all_latencies) / len(all_latencies)) if all_latencies else 0
    total_delay  = avg_lat_s * num_streams

    print()
    print("*" * 66)
    print(f"  AGGREGATE - {num_streams} stream(s)")
    print("*" * 66)
    print(f"  {'Total frames (all streams)':<38} {agg_stats['frames']:>18}")
    print(f"  {'Wall-clock time':<38} {total_wall:>17.3f}s")
    print(f"  {'Avg latency / frame':<38} {agg_stats['avg_ms']:>16.2f}ms")
    print(f"  {'Min latency':<38} {agg_stats['min_ms']:>16.2f}ms")
    print(f"  {'Max latency':<38} {agg_stats['max_ms']:>16.2f}ms")
    print(f"  {'Median (P50)':<38} {agg_stats['median_ms']:>16.2f}ms")
    print(f"  {'P95 latency':<38} {agg_stats['p95_ms']:>16.2f}ms")
    print(f"  {'P99 latency':<38} {agg_stats['p99_ms']:>16.2f}ms")
    print(f"  {'Aggregate throughput (FPS)':<38} {agg_stats['fps']:>18.2f}")
    print(f"  {'Total delay (avg_lat - N)':<38} {total_delay * 1000:>15.2f}ms")
    print(f"  {'Threshold (5 000 ms)':<38} {'- PASS' if total_delay <= MULTI_THRESHOLD else '- FAIL':>18}")
    print("*" * 66)

    return {
        "num_streams":    num_streams,
        "per_stream":     per_stream_stats,
        "aggregate":      agg_stats,
        "wall_time_s":    total_wall,
        "total_delay_s":  total_delay,
        "within_threshold": total_delay <= MULTI_THRESHOLD,
    }


# -
# Probe - quick measurement for auto-scaling
# -
def probe_multi_stream(video_paths, num_streams, probe_frames=PROBE_FRAMES, display=False):
    """
    Quick probe: run *num_streams* parallel workers for *probe_frames* each
    and return the estimated total delay.
    """
    print(f"\n  - Probing {num_streams} stream(s) ({probe_frames} frames each)...", end=" ", flush=True)
    stream_videos = [random.choice(video_paths) for _ in range(num_streams)]

    results_dict = {}
    lock = threading.Lock()

    display_frames = {} if display else None
    display_lock = threading.Lock() if display else None

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_streams) as pool:
        futures = []
        for sid in range(1, num_streams + 1):
            fut = pool.submit(
                _stream_worker,
                sid,
                stream_videos[sid - 1],
                probe_frames,
                results_dict,
                lock,
                display,
                display_frames,
                display_lock
            )
            futures.append(fut)
            
        if display:
            while not all(f.done() for f in futures):
                with display_lock:
                    grid = create_grid_image(display_frames, num_streams)
                if grid is not None and grid.size > 0:
                    cv2.imshow("Probe Benchmark", grid)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            cv2.destroyWindow("Probe Benchmark")
            cv2.waitKey(1)
            
        for fut in as_completed(futures):
            fut.result()
    wall_end = time.perf_counter()
    
    gc.collect()

    all_latencies = []
    for sid in sorted(results_dict.keys()):
        all_latencies.extend(results_dict[sid]["latencies"])

    avg_lat_s   = (sum(all_latencies) / len(all_latencies)) if all_latencies else 0
    total_delay = avg_lat_s * num_streams
    fps = len(all_latencies) / (wall_end - wall_start) if (wall_end - wall_start) > 0 else 0

    print(
        f"avg_lat={avg_lat_s*1000:.1f}ms  "
        f"total_delay={total_delay*1000:.0f}ms  "
        f"FPS={fps:.1f}  "
        f"{'-' if total_delay <= MULTI_THRESHOLD else '-'}"
    )
    return total_delay, avg_lat_s


# -
# Auto-scale: discover max cameras within 5 s threshold
# -
def auto_scale_benchmark(video_paths, max_frames=None, display=False):
    """
    1. Probe N = 1, 2, 3, - until total_delay > MULTI_THRESHOLD.
    2. Run full benchmarks for every valid N.

    Returns
    -------
    list of result dicts (one per camera count).
    """
    print("\n" + "-" * 66)
    print("  AUTO-SCALE: Discovering maximum camera count (threshold = "
          f"{MULTI_THRESHOLD:.0f}s)")
    print("-" * 66)

    max_n = 0

    # Phase 1 - probing
    for n in range(1, 100):  # safety cap at 99
        total_delay, _ = probe_multi_stream(video_paths, n, PROBE_FRAMES, display=display)
        if total_delay > MULTI_THRESHOLD:
            print(f"\n  - {n} stream(s): total delay {total_delay*1000:.0f}ms > "
                  f"{MULTI_THRESHOLD*1000:.0f}ms - stopping.")
            break
        max_n = n

    if max_n == 0:
        print("\n  Even 1 stream exceeds the 5-second threshold!")
        return []

    print(f"\n  - Maximum cameras within threshold: {max_n}")
    print("=" * 66)

    # Phase 2 - full benchmarks for N = 1 - max_n
    results = []
    for n in range(1, max_n + 1):
        print(f"\n{'-' * 66}")
        print(f"  FULL BENCHMARK - {n} stream(s)")
        print(f"{'-' * 66}")
        res = benchmark_multi_stream(video_paths, n, max_frames=max_frames, display=display)
        results.append(res)

    return results


# -
# Report generation
# -
def generate_report(
    single_results,
    multi_results,
    max_cameras,
    output_path,
    input_folder,
    display_enabled=False,
):
    """Write a Markdown benchmark report to *output_path*."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    L = lines.append

    L("# PeopleNet Detection - Benchmark Report")
    L("")
    L(f"**Generated**: {now}  ")
    L(f"**Input folder**: `{input_folder}`  ")
    L("")

    # - System information -
    L("## System Information")
    L("")
    L(f"| Property | Value |")
    L(f"|---|---|")
    L(f"| OS | {platform.system()} {platform.release()} ({platform.machine()}) |")
    L(f"| Python | {platform.python_version()} |")
    L(f"| PyTorch | {torch.__version__} |")
    L(f"| CUDA available | {torch.cuda.is_available()} |")
    if torch.cuda.is_available():
        L(f"| GPU | {torch.cuda.get_device_name(0)} |")
        L(f"| CUDA version | {torch.version.cuda} |")
    import onnxruntime as ort
    L(f"| ONNX Runtime | {ort.__version__} |")
    L(f"| ORT providers | {', '.join(peoplenet.session.get_providers())} |")
    L(f"| Benchmark cap | {BENCHMARK_CAP_S:.0f}s per run |")
    L(f"| Warmup frames | {WARMUP_FRAMES} |")
    L(f"| Infer every N | {INFER_EVERY_N} |")
    L(f"| Multi-stream threshold | {MULTI_THRESHOLD:.0f}s |")
    L(f"| Real-time Display | {'ON' if display_enabled else 'OFF'} |")
    L("")

    # - Single-stream results -
    if single_results:
        L("---")
        L("")
        L("## Single-Stream Benchmark Results")
        L("")
        L("| Video | Frames | Wall Time (s) | Avg (ms) | Min (ms) | Max (ms) | "
          "Median (ms) | P95 (ms) | P99 (ms) | FPS |")
        L("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for res in single_results:
            s = res["stats"]
            name = os.path.basename(res["video"])
            L(f"| {name} | {s['frames']} | {s['wall_time_s']:.2f} | "
              f"{s['avg_ms']:.2f} | {s['min_ms']:.2f} | {s['max_ms']:.2f} | "
              f"{s['median_ms']:.2f} | {s['p95_ms']:.2f} | {s['p99_ms']:.2f} | "
              f"{s['fps']:.2f} |")
        L("")

    # - Multi-stream results -
    if multi_results:
        L("---")
        L("")
        L("## Multi-Stream Benchmark Results")
        L("")
        L(f"**Maximum cameras within {MULTI_THRESHOLD:.0f}s threshold**: "
          f"**{max_cameras}**")
        L("")

        # Summary table
        L("### Scaling Summary")
        L("")
        L("| Cameras | Frames | Wall Time (s) | Avg Latency (ms) | "
          "Total Delay (ms) | Agg FPS | Status |")
        L("|---:|---:|---:|---:|---:|---:|---|")
        for res in multi_results:
            agg = res["aggregate"]
            n   = res["num_streams"]
            td  = res["total_delay_s"] * 1000
            ok  = "- PASS" if res["within_threshold"] else "- FAIL"
            L(f"| {n} | {agg['frames']} | {res['wall_time_s']:.2f} | "
              f"{agg['avg_ms']:.2f} | {td:.0f} | {agg['fps']:.2f} | {ok} |")
        L("")

        # Per-stream detail for each camera count
        for res in multi_results:
            n = res["num_streams"]
            L(f"### {n}-Camera Detail")
            L("")
            L("| Stream | Video | Frames | Avg (ms) | P95 (ms) | FPS |")
            L("|---:|---|---:|---:|---:|---:|")
            for sid in sorted(res["per_stream"].keys()):
                ps = res["per_stream"][sid]
                s  = ps["stats"]
                vn = os.path.basename(ps["video"])
                L(f"| {sid} | {vn} | {s['frames']} | {s['avg_ms']:.2f} | "
                  f"{s['p95_ms']:.2f} | {s['fps']:.2f} |")
            L("")

        # Scaling analysis
        L("### Scaling Analysis")
        L("")
        if len(multi_results) >= 2:
            base_fps = multi_results[0]["aggregate"]["fps"]
            L("| Cameras | Agg FPS | Efficiency vs 1-cam |")
            L("|---:|---:|---:|")
            for res in multi_results:
                n   = res["num_streams"]
                fps = res["aggregate"]["fps"]
                eff = (fps / base_fps * 100) if base_fps > 0 else 0
                L(f"| {n} | {fps:.2f} | {eff:.1f}% |")
            L("")
        L(f"> **Conclusion**: The system supports up to **{max_cameras}** "
          f"concurrent camera stream(s) while keeping the total per-cycle "
          f"delay below {MULTI_THRESHOLD:.0f} seconds.")
        L("")

    # - Write -
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  - Benchmark report saved to: {output_path}")


# -
# Interactive entry-point
# -
def main():
    print("\n" + "-" * 66)
    print("  PeopleNet Detection - Benchmark Mode")
    print("-" * 66)

    # - Input folder -
    input_folder = input("\nEnter input video folder path: ").strip()
    if not input_folder or not os.path.isdir(input_folder):
        print(f"Error: folder not found - {input_folder}")
        sys.exit(1)

    videos = discover_videos(input_folder)
    if not videos:
        print(f"Error: no video files found in {input_folder}")
        sys.exit(1)

    print(f"\n  Found {len(videos)} video(s):")
    for v in videos:
        print(f"    - {v}")

    # - Max frames -
    mf_input = input(
        "\nMax frames per benchmark run (Enter for unlimited, cap 60s): "
    ).strip()
    max_frames = int(mf_input) if mf_input.isdigit() else None

    # - Display option -
    disp_input = input(
        "\nDisplay real-time video? [y/N]: "
    ).strip().lower()
    display_enabled = disp_input in ['y', 'yes']

    # -
    # Phase 1 - Single-stream benchmarks (all videos)
    # -
    print("\n" + "-" * 66)
    print("  PHASE 1 : SINGLE-STREAM BENCHMARKS")
    print("-" * 66)

    single_results = []
    for vp in videos:
        res = benchmark_single_stream(vp, max_frames=max_frames, display=display_enabled)
        single_results.append(res)

    # -
    # Phase 2 - Multi-stream auto-scale benchmarks
    # -
    print("\n" + "-" * 66)
    print("  PHASE 2 : MULTI-STREAM BENCHMARKS (auto-scale)")
    print("-" * 66)

    multi_results = auto_scale_benchmark(videos, max_frames=max_frames, display=display_enabled)
    max_cameras = multi_results[-1]["num_streams"] if multi_results else 0

    # -
    # Phase 3 - Generate report
    # -
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"benchmark_report_{timestamp}.md"
    report_path = os.path.join(input_folder, report_name)

    generate_report(
        single_results=single_results,
        multi_results=multi_results,
        max_cameras=max_cameras,
        output_path=report_path,
        input_folder=input_folder,
        display_enabled=display_enabled,
    )

    print("\n" + "-" * 66)
    print("  BENCHMARK COMPLETE")
    print("-" * 66)
    print(f"  Report : {report_path}")
    print(f"  Max cameras within {MULTI_THRESHOLD:.0f}s threshold : {max_cameras}")
    print("-" * 66)


if __name__ == "__main__":
    main()
