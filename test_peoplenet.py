#!/usr/bin/env python
"""Quick test of PeopleNet video processing."""
from PeopleNetDet import load_peoplenet_model, process_video, DEVICE

print(f"Using device: {DEVICE}")
print("Loading PeopleNet model...")
model = load_peoplenet_model()
print(f"Model input shape: {model.get_inputs()[0].shape}")
print(f"ONNX providers: {model.get_providers()}")

video_path = r"F:\Other\Internship-Neelaminds\688-10_l.mov"
output_path = r"F:\Other\Internship-Neelaminds\688-10_l_marked_people_test.mp4"

print(f"\nProcessing video: {video_path}")
print(f"Output: {output_path}\n")
process_video(video_path, output_path=output_path, model=model, model_type="people")
print("\nDone!")
