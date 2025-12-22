import os
import torch
VIDEO_DIR = "Videos"
OUTPUT_DIR = "output"
TEMP_DIR = "temp_processed"
LOGO_PATH = "l&h_logo.png"
ANALYSIS_JSON = os.path.join(OUTPUT_DIR, "video_analysis.json")
FINAL_VIDEO = os.path.join(OUTPUT_DIR, "final_output.mp4")
FINAL_FPS = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANALYSIS_SAMPLE_RATE = 5
BATCH_SIZE = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
