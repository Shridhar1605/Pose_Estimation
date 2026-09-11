import os
import time
import threading
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
# dataset link: https://www.kaggle.com/datasets/fmena14/crowd-counting
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
model = YOLO("yolov8x.pt")
if hasattr(model, "to"):
    model.to(DEVICE)
def detect_persons(source):
    results = model(source, classes=[0], device=DEVICE)  # class 0 = person
    return results
def split_image_into_quadrants(image):
    h, w = image.shape[:2]
    mid_x, mid_y = w // 2, h // 2
    quadrants = [
        (0, 0, mid_x, mid_y, image[0:mid_y, 0:mid_x]),
        (mid_x, 0, w - mid_x, mid_y, image[0:mid_y, mid_x:w]),
        (0, mid_y, mid_x, h - mid_y, image[mid_y:h, 0:mid_x]),
        (mid_x, mid_y, w - mid_x, h - mid_y, image[mid_y:h, mid_x:w]),
    ]
    return quadrants
def stitch_quadrants(quadrants, image_shape):
    h, w = image_shape[:2]
    stitched = np.zeros(image_shape, dtype=np.uint8)
    for x, y, width, height, quad in quadrants:
        stitched[y:y + height, x:x + width] = quad
    return stitched
def process_image_with_quadrants(source_path, output_path):
    image = cv2.imread(source_path)
    if image is None:
        raise ValueError(f"Unable to read image: {source_path}")
    quadrants = split_image_into_quadrants(image)
    marked_quads = []
    for x, y, width, height, quad in quadrants:
        print(f"Processing quadrant at ({x}, {y}) for {source_path}...")
        results = detect_persons(quad)
        marked_quad = results[0].plot()
        marked_quads.append((x, y, width, height, marked_quad))
    stitched = stitch_quadrants(marked_quads, image.shape)
    cv2.imwrite(output_path, stitched)
    print(f"Saved stitched image {output_path}")
def process_images(input_dir="images", output_dir="marked_images", split_to_quadrants=False):
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for root, _, files in os.walk(input_dir):
        rel_root = os.path.relpath(root, input_dir)
        target_root = os.path.join(output_dir, rel_root) if rel_root != "." else output_dir
        os.makedirs(target_root, exist_ok=True)
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in valid_exts:
                continue
            source_path = os.path.join(root, file_name)
            output_path = os.path.join(target_root, file_name)
            print(f"Processing {source_path}...")
            if split_to_quadrants:
                process_image_with_quadrants(source_path, output_path)
            else:
                results = detect_persons(source_path)
                marked = results[0].plot()
                cv2.imwrite(output_path, marked)
                print(f"Saved {output_path}")
def process_frame(frame, split_to_quadrants=False):
    if not split_to_quadrants:
        results = detect_persons(frame)
        marked = results[0].plot()
        return marked
    quadrants = split_image_into_quadrants(frame)
    marked_quads = []
    for x, y, width, height, quad in quadrants:
        results = detect_persons(quad)
        marked_quad = results[0].plot()
        marked_quads.append((x, y, width, height, marked_quad))
    stitched = stitch_quadrants(marked_quads, frame.shape)
    return stitched
def process_videos(input_dir="video", output_dir="marked_videos", split_to_quadrants=False):
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}
    for root, _, files in os.walk(input_dir):
        rel_root = os.path.relpath(root, input_dir)
        target_root = os.path.join(output_dir, rel_root) if rel_root != "." else output_dir
        os.makedirs(target_root, exist_ok=True)
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in valid_exts:
                continue
            source_path = os.path.join(root, file_name)
            output_path = os.path.join(target_root, file_name)
            print(f"Processing video {source_path}...")
            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                print(f"Failed to open {source_path}")
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                marked = process_frame(frame, split_to_quadrants=split_to_quadrants)
                if marked is None:
                    marked = frame
                if marked.shape[1] != width or marked.shape[0] != height:
                    marked = cv2.resize(marked, (width, height))
                out.write(marked)
                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx} frames for {file_name}...")
            cap.release()
            out.release()
            print(f"Saved marked video {output_path}")
# ─────────────────────────────────────────────────────────────
# Benchmarking Mode
# ─────────────────────────────────────────────────────────────
def _compute_percentile(sorted_values, percentile):
    """Compute the given percentile from an already-sorted list."""
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * percentile / 100.0)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]
def _print_latency_report(title, latencies, total_time):
    """Pretty-print a latency summary table."""
    if not latencies:
        print(f"\n{title}: No frames processed.")
        return
    sorted_lat = sorted(latencies)
    avg = sum(sorted_lat) / len(sorted_lat)
    fps = len(sorted_lat) / total_time if total_time > 0 else 0.0
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)
    print(f"  {'Metric':<30} {'Value':>20}")
    print("-" * 62)
    print(f"  {'Frames processed':<30} {len(sorted_lat):>20}")
    print(f"  {'Total wall-clock time':<30} {total_time:>19.3f}s")
    print(f"  {'Avg latency / frame':<30} {avg * 1000:>18.2f}ms")
    print(f"  {'Min latency':<30} {sorted_lat[0] * 1000:>18.2f}ms")
    print(f"  {'Max latency':<30} {sorted_lat[-1] * 1000:>18.2f}ms")
    print(f"  {'Median (P50)':<30} {_compute_percentile(sorted_lat, 50) * 1000:>18.2f}ms")
    print(f"  {'P95 latency':<30} {_compute_percentile(sorted_lat, 95) * 1000:>18.2f}ms")
    print(f"  {'P99 latency':<30} {_compute_percentile(sorted_lat, 99) * 1000:>18.2f}ms")
    print(f"  {'Throughput (FPS)':<30} {fps:>20.2f}")
    print("=" * 62)
def benchmark_single_stream(video_path, split_to_quadrants=False, max_frames=None):
    """
    Benchmark inference latency on a single video stream.
    Args:
        video_path: Path to the input video file.
        split_to_quadrants: Whether to split each frame into 4 quadrants.
        max_frames: If set, stop after processing this many frames.
    Returns:
        A dict with latency stats and the per-frame latency list.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = max_frames if max_frames else total_frames
    print(f"\n  Video       : {video_path}")
    print(f"  Resolution  : {width}x{height} @ {vid_fps:.1f} FPS")
    print(f"  Total frames: {total_frames}  (benchmarking up to {limit})")
    print(f"  Quadrants   : {'Yes' if split_to_quadrants else 'No'}")
    print()
    latencies = []
    frame_idx = 0
    warmup_frames = 5  # first few frames are warmup (not counted)
    wall_start = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        if max_frames and frame_idx >= max_frames + warmup_frames:
            break
        if time.perf_counter() - wall_start >= 60.0:
            break
        t0 = time.perf_counter()
        _ = process_frame(frame, split_to_quadrants=split_to_quadrants)
        t1 = time.perf_counter()
        if frame_idx >= warmup_frames:
            latencies.append(t1 - t0)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"    [{os.path.basename(video_path)}] Processed {frame_idx} frames...")
    wall_end = time.perf_counter()
    cap.release()
    total_time = wall_end - wall_start
    _print_latency_report(
        f"Single-Stream Benchmark — {os.path.basename(video_path)}",
        latencies,
        total_time,
    )
    return {
        "video": video_path,
        "frames": len(latencies),
        "total_time_s": total_time,
        "latencies": latencies,
    }
def _stream_worker(stream_id, video_path, split_to_quadrants, max_frames, results_dict, lock):
    """
    Worker function for a single stream inside the multi-stream benchmark.
    Reads frames from its own VideoCapture and records per-frame latencies.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [Stream {stream_id}] ERROR: Cannot open {video_path}")
        return
    warmup_frames = 5
    latencies = []
    frame_idx = 0
    worker_start = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        if max_frames and frame_idx >= max_frames + warmup_frames:
            break
        if time.perf_counter() - worker_start >= 60.0:
            break
        t0 = time.perf_counter()
        _ = process_frame(frame, split_to_quadrants=split_to_quadrants)
        t1 = time.perf_counter()
        if frame_idx >= warmup_frames:
            latencies.append(t1 - t0)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"    [Stream {stream_id}] Processed {frame_idx} frames...")
    cap.release()
    with lock:
        results_dict[stream_id] = latencies
def benchmark_multi_stream(video_path, num_streams, split_to_quadrants=False, max_frames=None):
    """
    Benchmark inference latency across multiple concurrent video streams.
    The same video is duplicated `num_streams` times, each read by its own
    thread to simulate independent camera feeds.
    Args:
        video_path: Path to the input video file.
        num_streams: Number of concurrent streams (threads).
        split_to_quadrants: Whether to split each frame into 4 quadrants.
        max_frames: If set, each stream stops after this many frames.
    """
    cap_check = cv2.VideoCapture(video_path)
    if not cap_check.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap_check.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_check.release()
    limit = max_frames if max_frames else total_frames
    print(f"\n  Video        : {video_path}")
    print(f"  Resolution   : {width}x{height} @ {vid_fps:.1f} FPS")
    print(f"  Total frames : {total_frames}  (benchmarking up to {limit} per stream)")
    print(f"  Quadrants    : {'Yes' if split_to_quadrants else 'No'}")
    print(f"  Streams      : {num_streams}")
    print()
    results_dict = {}
    lock = threading.Lock()
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_streams) as pool:
        futures = []
        for sid in range(1, num_streams + 1):
            fut = pool.submit(
                _stream_worker,
                sid,
                video_path,
                split_to_quadrants,
                max_frames,
                results_dict,
                lock,
            )
            futures.append(fut)
        # Wait for all to finish
        for fut in as_completed(futures):
            fut.result()  # re-raise any exceptions
    wall_end = time.perf_counter()
    total_wall = wall_end - wall_start
    # ── Per-stream reports ──────────────────────────────────
    all_latencies = []
    for sid in sorted(results_dict.keys()):
        lats = results_dict[sid]
        stream_total = sum(lats) if lats else 0.0
        _print_latency_report(f"Stream {sid}", lats, stream_total)
        all_latencies.extend(lats)
    # ── Aggregate report ────────────────────────────────────
    print()
    print("*" * 62)
    print(f"  AGGREGATE — {num_streams} streams")
    print("*" * 62)
    if all_latencies:
        sorted_all = sorted(all_latencies)
        avg = sum(sorted_all) / len(sorted_all)
        total_frames_processed = len(sorted_all)
        agg_fps = total_frames_processed / total_wall if total_wall > 0 else 0.0
        print(f"  {'Total frames (all streams)':<35} {total_frames_processed:>15}")
        print(f"  {'Wall-clock time':<35} {total_wall:>14.3f}s")
        print(f"  {'Avg latency / frame':<35} {avg * 1000:>13.2f}ms")
        print(f"  {'Min latency':<35} {sorted_all[0] * 1000:>13.2f}ms")
        print(f"  {'Max latency':<35} {sorted_all[-1] * 1000:>13.2f}ms")
        print(f"  {'Median (P50)':<35} {_compute_percentile(sorted_all, 50) * 1000:>13.2f}ms")
        print(f"  {'P95 latency':<35} {_compute_percentile(sorted_all, 95) * 1000:>13.2f}ms")
        print(f"  {'P99 latency':<35} {_compute_percentile(sorted_all, 99) * 1000:>13.2f}ms")
        print(f"  {'Aggregate throughput (FPS)':<35} {agg_fps:>15.2f}")
    else:
        print("  No frames were processed.")
    print("*" * 62)
def run_benchmark():
    """Interactive entry-point for the benchmarking mode."""
    video_path = input("Enter path to the video file for benchmarking: ").strip()
    if not os.path.isfile(video_path):
        print(f"Error: File not found — {video_path}")
        return
    quad_choice = input("Split frames into quadrants? [keep/split] (default: keep): ").strip().lower()
    split_flag = quad_choice == "split"
    max_frames_input = input("Max frames to benchmark per stream (press Enter for all): ").strip()
    max_frames = int(max_frames_input) if max_frames_input.isdigit() else None
    stream_mode = input("Single stream or multi stream? [single/multi] (default: single): ").strip().lower()
    if stream_mode == "multi":
        num_input = input("How many streams? ").strip()
        if not num_input.isdigit() or int(num_input) < 1:
            print("Invalid number of streams, defaulting to 2.")
            num_streams = 2
        else:
            num_streams = int(num_input)
        print(f"\n{'─' * 62}")
        print(f"  Starting MULTI-STREAM benchmark  ({num_streams} streams)")
        print(f"{'─' * 62}")
        benchmark_multi_stream(
            video_path,
            num_streams,
            split_to_quadrants=split_flag,
            max_frames=max_frames,
        )
    else:
        print(f"\n{'─' * 62}")
        print(f"  Starting SINGLE-STREAM benchmark")
        print(f"{'─' * 62}")
        benchmark_single_stream(
            video_path,
            split_to_quadrants=split_flag,
            max_frames=max_frames,
        )
if __name__ == "__main__":
    mode = input(
        "Choose mode — [images / videos / benchmark] (default: videos): "
    ).strip().lower()
    if mode not in {"images", "videos", "benchmark", ""}:
        print("Invalid choice, defaulting to videos.")
        mode = "videos"
    if mode == "benchmark":
        run_benchmark()
    elif mode == "images":
        choice = input("Process images as full images or split into 4 parts? [keep/split]: ").strip().lower()
        split_flag = choice == "split"
        if choice not in {"keep", "split"}:
            print("Invalid choice, defaulting to keep full images.")
        process_images(split_to_quadrants=split_flag)
    else:
        choice = input("Process video frames as full frames or split into 4 parts? [keep/split]: ").strip().lower()
        split_flag = choice == "split"
        if choice not in {"keep", "split", ""}:
            print("Invalid choice, defaulting to keep full frames.")
        process_videos(split_to_quadrants=split_flag)