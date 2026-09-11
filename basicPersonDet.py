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
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_persons(frame)
        marked = results[0].plot()
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
