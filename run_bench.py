import sys
import os

sys.path.append(r'f:\Other\Internship-Neelaminds')
import personDetection as pd

video_path = r'f:\Other\Internship-Neelaminds\Video_samples\first\688-10_l.mov'

print("Starting 1-minute Single Stream Benchmark...")
pd.benchmark_single_stream(video_path, split_to_quadrants=False, max_frames=None)

print("\nStarting 1-minute Multi-Stream Benchmark (2 streams)...")
pd.benchmark_multi_stream(video_path, num_streams=2, split_to_quadrants=False, max_frames=None)
