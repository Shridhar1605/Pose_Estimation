import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from boxmot.trackers.bbox.ocsort.ocsort import OcSort
import torchvision

#dataset link: https://www.kaggle.com/datasets/fmena14/crowd-counting
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

model = YOLO("yolo26n.pt")
model.to(DEVICE)  # or just pass device during inference

def detect_persons_robust(image):
    H, W = image.shape[:2]
    all_dets = []
    
    # Normal inference
    res0 = model.predict(image, classes=[0], device=DEVICE, verbose=False, conf=0.2, imgsz=1280)[0]
    boxes0 = res0.boxes.xyxy.cpu().numpy()
    confs0 = res0.boxes.conf.cpu().numpy().reshape(-1, 1)
    if len(boxes0) > 0:
        all_dets.append(np.concatenate([boxes0, confs0], axis=1))
        
    # CW inference
    img_cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    res_cw = model.predict(img_cw, classes=[0], device=DEVICE, verbose=False, conf=0.2, imgsz=1280)[0]
    boxes_cw = res_cw.boxes.xyxy.cpu().numpy()
    confs_cw = res_cw.boxes.conf.cpu().numpy().reshape(-1, 1)
    dets_cw = []
    for (x1_p, y1_p, x2_p, y2_p), conf in zip(boxes_cw, confs_cw):
        x1 = y1_p
        y1 = H - x2_p
        x2 = y2_p
        y2 = H - x1_p
        dets_cw.append([x1, y1, x2, y2, conf[0]])
    if dets_cw:
        all_dets.append(np.array(dets_cw))
    
    # CCW inference
    img_ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    res_ccw = model.predict(img_ccw, classes=[0], device=DEVICE, verbose=False, conf=0.2, imgsz=1280)[0]
    boxes_ccw = res_ccw.boxes.xyxy.cpu().numpy()
    confs_ccw = res_ccw.boxes.conf.cpu().numpy().reshape(-1, 1)
    dets_ccw = []
    for (x1_p, y1_p, x2_p, y2_p), conf in zip(boxes_ccw, confs_ccw):
        x1 = W - y2_p
        y1 = x1_p
        x2 = W - y1_p
        y2 = x2_p
        dets_ccw.append([x1, y1, x2, y2, conf[0]])
    if dets_ccw:
        all_dets.append(np.array(dets_ccw))
    
    if len(all_dets) == 0:
        return np.empty((0, 5))
        
    all_dets = np.vstack(all_dets)
    
    # NMS
    boxes_t = torch.tensor(all_dets[:, :4], dtype=torch.float32)
    scores_t = torch.tensor(all_dets[:, 4], dtype=torch.float32)
    keep = torchvision.ops.nms(boxes_t, scores_t, 0.45)
    return all_dets[keep.numpy()]


class OCSortTracker:
    def __init__(self, iou_threshold=0.3, max_lost=30, min_confidence=0.3):
        self.tracker = OcSort(
            det_thresh=min_confidence,
            max_age=max_lost,
            min_hits=1,
            iou_threshold=iou_threshold,
            per_class=False
        )

    def update(self, detections, frame=None):
        if len(detections) == 0:
            dets_np = np.empty((0, 6), dtype=np.float32)
        else:
            dets_np = np.array(detections, dtype=np.float32)
            cls_col = np.zeros((dets_np.shape[0], 1), dtype=np.float32)
            dets_np = np.concatenate([dets_np, cls_col], axis=1)

        if frame is None:
            frame = np.zeros((100, 100, 3), dtype=np.uint8)

        res = self.tracker.update(dets_np, frame)
        
        tracked = []
        for r in res:
            x1, y1, x2, y2, track_id, conf, cls, ind = r
            tracked.append({
                "id": int(track_id),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(conf)
            })
        return tracked


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
        dets = detect_persons_robust(quad)
        marked_quad = quad.copy()
        for d in dets:
            x1, y1, x2, y2, conf = d
            cv2.rectangle(marked_quad, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(marked_quad, f"{conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        marked_quads.append((x, y, width, height, marked_quad))

    stitched = stitch_quadrants(marked_quads, image.shape)
    cv2.imwrite(output_path, stitched)
    print(f"Saved stitched image {output_path}")


def process_images(input_dir="images", output_dir="marked_images", split_to_quadrants=False):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        abs_root = os.path.abspath(root)
        # Prevent recursively processing files written to the output folder if it lives inside the input folder
        dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(abs_root, d)) != output_dir]
        if abs_root == output_dir or abs_root.startswith(output_dir + os.sep):
            continue

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

            start_time = time.time()

            if split_to_quadrants:
                process_image_with_quadrants(source_path, output_path)
            else:
                image = cv2.imread(source_path)
                if image is not None:
                    dets = detect_persons_robust(image)
                    marked = image.copy()
                    for d in dets:
                        x1, y1, x2, y2, conf = d
                        cv2.rectangle(marked, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(marked, f"{conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imwrite(output_path, marked)
            
            end_time = time.time()
            latency = end_time - start_time
            print(f"Saved {output_path}")
            print(f"Latency: {latency:.4f} seconds")


def process_videos(input_dir="videos", output_dir="marked_videos", display=False):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".ts", ".flv"}

    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        abs_root = os.path.abspath(root)
        dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(abs_root, d)) != output_dir]
        if abs_root == output_dir or abs_root.startswith(output_dir + os.sep):
            continue

        rel_root = os.path.relpath(root, input_dir)
        target_root = os.path.join(output_dir, rel_root) if rel_root != "." else output_dir
        os.makedirs(target_root, exist_ok=True)

        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in video_exts:
                continue

            source_path = os.path.join(root, file_name)
            base_name = os.path.splitext(file_name)[0]
            output_file_name = f"{base_name}_marked.mp4"
            output_path = os.path.join(target_root, output_file_name)
            print(f"Processing video {source_path}...")
            process_video(source_path, output_path=output_path, display=display)


def process_video(source_path, output_path=None, display=False):
    if output_path is None:
        base, _ = os.path.splitext(source_path)
        output_path = f"{base}_marked.mp4"

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    # Process YOLO detection every N frames; tracker Kalman-steps on all frames
    detect_every_n = 2
    tracker = OCSortTracker(iou_threshold=0.3, max_lost=30, min_confidence=0.3)
    total_processing_time = 0.0
    tracked = []

    # MOG2 background subtractor — the "memory" layer.
    # Builds a running model of the empty scene over ~100 frames so it can
    # highlight foreground (moving) regions cheaply on every frame.
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=100,          # frames used to learn the background
        varThreshold=50,      # sensitivity: lower = more sensitive to motion
        detectShadows=False   # skip shadow classification for speed
    )
    # Morphology kernel for cleaning up the foreground mask
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    MIN_MOTION_AREA = 800     # px² — ignore tiny noise blobs

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Always feed every frame into MOG2 so the background model stays
        # current (this is the temporal memory — it remembers the scene).
        fg_mask = bg_subtractor.apply(frame)

        if frame_count % detect_every_n == 0:
            # ── Detection frame: full YOLO inference ──────────────────────
            dets = detect_persons_robust(frame)
            if len(dets) > 0:
                tracked = tracker.update(dets.tolist(), frame)
            else:
                tracked = tracker.update([], frame)
        else:
            # ── Skipped frame: MOG2-guided ROI detection ──────────────────
            # Clean the foreground mask: remove noise, merge nearby blobs
            fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, morph_kernel)
            fg_clean = cv2.dilate(fg_clean, morph_kernel, iterations=3)

            contours, _ = cv2.findContours(
                fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Kalman-predicted bounding boxes of currently tracked persons
            predicted_boxes = [t["bbox"] for t in tracked]

            roi_dets = []
            for cnt in contours:
                if cv2.contourArea(cnt) < MIN_MOTION_AREA:
                    continue  # ignore noise

                mx, my, mw, mh = cv2.boundingRect(cnt)

                # Check whether this motion blob is already explained by an
                # existing Kalman-predicted track (overlap > 30 %)
                covered = False
                for pb in predicted_boxes:
                    px1, py1, px2, py2 = pb
                    ix1, iy1 = max(mx, px1), max(my, py1)
                    ix2, iy2 = min(mx + mw, px2), min(my + mh, py2)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    if mw * mh > 0 and inter / (mw * mh) > 0.3:
                        covered = True
                        break

                if covered:
                    continue  # Kalman already accounts for this region

                # Unexpected motion → run YOLO only on this small crop
                pad = 20
                rx1 = max(0, mx - pad)
                ry1 = max(0, my - pad)
                rx2 = min(width, mx + mw + pad)
                ry2 = min(height, my + mh + pad)
                roi = frame[ry1:ry2, rx1:rx2]
                if roi.size == 0:
                    continue

                roi_dets_arr = detect_persons_robust(roi)
                if len(roi_dets_arr) > 0:
                    for d in roi_dets_arr:
                        x1, y1, x2, y2, conf = d
                        # Translate crop-relative coords back to full-frame
                        roi_dets.append([
                            float(x1) + rx1, float(y1) + ry1,
                            float(x2) + rx1, float(y2) + ry1,
                            float(conf)
                        ])

            # Update tracker: new ROI detections if any, else pure Kalman step
            tracked = tracker.update(roi_dets, frame)

        marked = frame.copy()
        for track in tracked:
            x1, y1, x2, y2 = map(int, track["bbox"])
            score = track["score"]
            track_id = track["id"]
            label = f"ID:{track_id} {score:.2f}"
            cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(marked, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        count_label = f"Persons: {len(tracked)}"
        cv2.putText(marked, count_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        writer.write(marked)

        end_time = time.time()
        total_processing_time += (end_time - start_time)

        frame_count += 1

        if display:
            cv2.imshow("Processed Video", marked)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    avg_latency = total_processing_time / frame_count if frame_count > 0 else 0
    print(f"Saved processed video to {output_path} ({frame_count} frames)")
    print(f"Avg latency per frame: {avg_latency:.4f} seconds")


if __name__ == "__main__":
    choice = input("Process images, videos, or a single video? [images/videos/video]: ").strip().lower()
    if choice == "video":
        source_path = input("Enter video path: ").strip()
        output_path = input("Enter output video path (leave blank to auto-generate): ").strip() or None
        process_video(source_path, output_path=output_path)
    elif choice == "videos":
        input_dir = input("Enter input video folder path: ").strip() or "videos"
        output_dir = input("Enter output video folder path: ").strip() or "marked_videos"
        display_choice = input("Display each video while processing? [y/N]: ").strip().lower()
        display_flag = display_choice in {"y", "yes"}

        if not os.path.isdir(input_dir):
            raise ValueError(f"Input folder does not exist: {input_dir}")

        process_videos(input_dir=input_dir, output_dir=output_dir, display=display_flag)
    else:
        input_dir = input("Enter input image folder path: ").strip() or "images"
        output_dir = input("Enter output image folder path: ").strip() or "marked_images"
        split_choice = input("Split each image into quadrants? [y/N]: ").strip().lower()
        split_flag = split_choice in {"y", "yes"}

        if not os.path.isdir(input_dir):
            raise ValueError(f"Input folder does not exist: {input_dir}")

        process_images(input_dir=input_dir, output_dir=output_dir, split_to_quadrants=split_flag)
