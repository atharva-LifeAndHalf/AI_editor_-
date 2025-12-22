import cv2
import numpy as np
import os
import shutil
import subprocess
import time
from blip_clip_2 import cosine
from video_analysis import segment_scenes, deduplicate_frames
from config import FINAL_FPS, LOGO_PATH
from filters import apply_filter
from transitions import crossfade, dissolve, fade_to_black
from advance import apply_stabilization, remove_background, remove_object, add_logo_overlay

# CHANGE: Use MJPG for the temporary render. 
# It is a universal 'dumb' codec that won't throw the "codec_id 27" error.
FOURCC_TEMP = cv2.VideoWriter_fourcc(*'MJPG')

def optimize_for_web(input_path):
    """
    CRITICAL: This takes the MJPG file and converts it to H.264 
    using the system FFmpeg (which supports codec_id 27).
    """
    temp_output = input_path.replace(".mp4", "_temp.mp4")
    try:
        # We use 'libx264' which is the proper H.264 encoder
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264', 
            '-pix_fmt', 'yuv420p', # Required for web browsers
            '-preset', 'ultrafast', 
            '-crf', '23', # Good balance of quality/size
            temp_output
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(temp_output, input_path)
    except Exception as e:
        print(f"Web optimization failed: {e}")

# ---------------- PROCESSING ----------------

def process_clip_to_file(clip_data, style, output_path):
    frames = clip_data["frames"]
    if not frames: return 0

    print(f"   🎬 Segmenting scenes...")
    scenes, all_embeddings = segment_scenes(frames, style["scene_threshold"])

    h, w, _ = frames[0].shape
    # CHANGE: Use FOURCC_TEMP (MJPG) here
    out = cv2.VideoWriter(output_path, FOURCC_TEMP, FINAL_FPS, (w, h))

    total_written = 0
    for scene in scenes:
        clean_frames = deduplicate_frames(scene, all_embeddings, style["frame_similarity"])
        for frame in clean_frames:
            filtered = apply_filter(frame, style["filter"])
            out.write(cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR))
            total_written += 1

    out.release()
    # No optimize_for_web here yet, we do it at the very end of combination
    print(f"   ✓ Wrote {total_written} frames")
    return total_written

def combine_videos_with_transitions(temp_files, style, output_path):
    if not temp_files: return

    print("\n🎬 Combining with transitions...")
    cap = cv2.VideoCapture(temp_files[0])
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # CHANGE: Use FOURCC_TEMP (MJPG) here
    out = cv2.VideoWriter(output_path, FOURCC_TEMP, FINAL_FPS, (w, h))
    last_frame = None

    for idx, temp_file in enumerate(temp_files):
        cap = cv2.VideoCapture(temp_file)
        first_frame_of_clip = True
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if idx > 0 and first_frame_of_clip and last_frame is not None:
                first_frame_of_clip = False
                t_type = style.get("transition", "none")
                t_frames = style.get("transition_frames", 10)
                if t_type == "crossfade": trans = crossfade(last_frame, frame_rgb, t_frames)
                elif t_type == "dissolve": trans = dissolve(last_frame, frame_rgb, t_frames)
                elif t_type == "fade_to_black": trans = fade_to_black(last_frame, frame_rgb, t_frames)
                else: trans = []
                for t in trans:
                    out.write(cv2.cvtColor(t, cv2.COLOR_RGB2BGR))
            out.write(frame)
            last_frame = frame_rgb
        cap.release()

    out.release()
    time.sleep(0.5)
    # CRITICAL: Now convert the final MJPG file to H.264
    optimize_for_web(output_path)

def apply_ai_edits(input_path, edits, output_path, roi=None):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    new_fps = fps * edits.get("speed", 1.0)
    # CHANGE: Use FOURCC_TEMP
    out = cv2.VideoWriter(output_path, FOURCC_TEMP, new_fps, (w, h))

    start_frame = int(edits.get("trim_start", 0) * fps)
    end_frame = total_frames - int(edits.get("trim_end", 0) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    prev_gray = None
    idx = start_frame

    while idx < end_frame:
        ret, frame = cap.read()
        if not ret: break
        if edits.get("stabilize"):
            if prev_gray is None: prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else: frame, prev_gray = apply_stabilization(prev_gray, frame)
        if edits.get("remove_bg"): frame = remove_background(frame)
        if edits.get("erase_object") and roi: frame = remove_object(frame, roi)
        if edits.get("filter") and edits["filter"] != "none":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_filtered = apply_filter(frame_rgb, edits["filter"])
            frame = cv2.cvtColor(frame_filtered, cv2.COLOR_RGB2BGR)
        if edits.get("add_logo"): frame = add_logo_overlay(frame, LOGO_PATH)
        out.write(frame)
        idx += 1

    cap.release()
    out.release()
    optimize_for_web(output_path)

def semantic_order(clips):
    if len(clips) <= 1: return clips
    ordered = [clips.pop(0)]
    while clips:
        last_emb = np.array(ordered[-1]["embedding"])
        best_idx = 0
        best_sim = -1
        for i, c in enumerate(clips):
            sim = cosine(last_emb, np.array(c["embedding"]))
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        ordered.append(clips.pop(best_idx))
    return ordered
