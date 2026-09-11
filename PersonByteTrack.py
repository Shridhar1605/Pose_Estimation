import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

#dataset link: https://www.kaggle.com/datasets/fmena14/crowd-counting
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

model = YOLO("yolo26n.pt")
model.to(DEVICE)  # or just pass device during inference

def detect_persons(source):
    results = model.predict(source, classes=[0], device=DEVICE, verbose=False)
    return results


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_center_distance(box_a, box_b):
    ax = (box_a[0] + box_a[2]) / 2.0
    ay = (box_a[1] + box_a[3]) / 2.0
    bx = (box_b[0] + box_b[2]) / 2.0
    by = (box_b[1] + box_b[3]) / 2.0
    return np.hypot(ax - bx, ay - by)


class ByteTrack:
    def __init__(self, iou_threshold=0.15, max_lost=90, min_confidence=0.3):
        self.iou_threshold = iou_threshold
        self.match_threshold = 0.03
        self.max_lost = max_lost
        self.min_confidence = min_confidence
        self.next_id = 1
        self.tracks = []

    def _match_score(self, track_bbox, det_bbox):
        iou = compute_iou(track_bbox, det_bbox)
        if iou >= self.iou_threshold:
            return 0.8 * iou + 0.2

        dist = compute_center_distance(track_bbox, det_bbox)
        track_w = track_bbox[2] - track_bbox[0]
        track_h = track_bbox[3] - track_bbox[1]
        max_dim = max(track_w, track_h, 1.0)
        if dist < max_dim * 0.5:
            norm = 1.0 - min(dist / (max_dim * 0.5), 1.0)
            return 0.1 + 0.4 * norm

        return 0.0

    def _match_detections(self, detections):
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        score_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                score_matrix[t_idx, d_idx] = self._match_score(track["bbox"], det[:4])

        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))

        while True:
            if score_matrix.size == 0:
                break
            t_idx, d_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
            if score_matrix[t_idx, d_idx] < self.match_threshold:
                break
            matches.append((t_idx, d_idx))
            score_matrix[t_idx, :] = -1
            score_matrix[:, d_idx] = -1
            if t_idx in unmatched_tracks:
                unmatched_tracks.remove(t_idx)
            if d_idx in unmatched_detections:
                unmatched_detections.remove(d_idx)

        return matches, unmatched_detections, unmatched_tracks

    def update(self, detections):
        detections = [det for det in detections if det[4] >= self.min_confidence]
        matches, unmatched_dets, unmatched_tracks = self._match_detections(detections)

        for t_idx, d_idx in matches:
            det = detections[d_idx]
            prev_bbox = np.array(self.tracks[t_idx]["bbox"], dtype=np.float32)
            new_bbox = np.array(det[:4], dtype=np.float32)
            smoothed_bbox = 0.8 * prev_bbox + 0.2 * new_bbox
            self.tracks[t_idx]["bbox"] = list(smoothed_bbox.tolist())
            self.tracks[t_idx]["score"] = float(det[4])
            self.tracks[t_idx]["lost"] = 0
            self.tracks[t_idx]["age"] = self.tracks[t_idx].get("age", 0) + 1

        for track_idx in unmatched_tracks:
            self.tracks[track_idx]["lost"] += 1
            self.tracks[track_idx]["age"] = self.tracks[track_idx].get("age", 0) + 1

        for d_idx in unmatched_dets:
            det = detections[d_idx]
            self.tracks.append({
                "id": self.next_id,
                "bbox": list(np.array(det[:4], dtype=np.float32).tolist()),
                "score": float(det[4]),
                "lost": 0,
                "age": 1,
            })
            self.next_id += 1

        self.tracks = [track for track in self.tracks if track["lost"] <= self.max_lost]
        return [track for track in self.tracks if track["lost"] == 0]


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

            if split_to_quadrants:
                process_image_with_quadrants(source_path, output_path)
            else:
                results = detect_persons(source_path)
                marked = results[0].plot()
                cv2.imwrite(output_path, marked)
                print(f"Saved {output_path}")


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
    tracker = ByteTrack(iou_threshold=0.3, max_lost=30, min_confidence=0.3)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_persons(frame)

        if len(results[0].boxes) > 0:
            dets = np.concatenate([
                results[0].boxes.xyxy.cpu().numpy(),
                results[0].boxes.conf.cpu().numpy().reshape(-1, 1),
            ], axis=1)
            tracked = tracker.update(dets.tolist())
        else:
            tracked = []

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
        frame_count += 1

        if display:
            cv2.imshow("Processed Video", marked)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    print(f"Saved processed video to {output_path} ({frame_count} frames)")


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
