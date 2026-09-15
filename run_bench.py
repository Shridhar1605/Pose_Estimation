import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import personDetection as pd

video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Video_samples', 'vtest_pedestrians.avi')

print("Starting 1-minute Single Stream Benchmark...")
pd.benchmark_single_stream(video_path, split_to_quadrants=False, max_frames=None)

print("\nStarting 1-minute Multi-Stream Benchmark (2 streams)...")
pd.benchmark_multi_stream(video_path, num_streams=2, split_to_quadrants=False, max_frames=None)
