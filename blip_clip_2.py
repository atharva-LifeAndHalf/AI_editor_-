import os
import cv2
import gc
import torch
import json
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration
)
from config import DEVICE, OUTPUT_DIR, VIDEO_DIR

# --- 2025 STABILITY PATCHES ---
# Prevents the "ReadTimeout" that busts the link during model download
os.environ["HF_HUB_READ_TIMEOUT"] = "120"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

# --- MEDIAPIPE INITIALIZATION ---
try:
    import mediapipe as mp
    from mediapipe.python.solutions import selfie_segmentation as mp_selfie_seg
    segmentor = mp_selfie_seg.SelfieSegmentation(model_selection=1)
    print("✅ MediaPipe initialized")
except Exception as e:
    print(f"⚠️ MediaPipe unavailable: {e}")
    segmentor = None

# ---------------- STYLES ----------------
STYLES = {
    "cinematic": {"scene_threshold": 0.70, "frame_similarity": 0.96, "transition": "crossfade", "transition_frames": 25, "filter": "cinematic"},
    "fast_reel": {"scene_threshold": 0.60, "frame_similarity": 0.93, "transition": "cut", "transition_frames": 0, "filter": "bright"},
    "documentary": {"scene_threshold": 0.80, "frame_similarity": 0.97, "transition": "crossfade", "transition_frames": 15, "filter": "none"},
    "vlog": {"scene_threshold": 0.65, "frame_similarity": 0.94, "transition": "cut", "transition_frames": 5, "filter": "none"}
}

# ---------------- LOAD MODELS (CACHED TO PREVENT RAM CRASH) ----------------
@st.cache_resource
def load_ai_models():
    """Loads models into memory once. Crucial for keeping the link alive."""
    print(f"🔹 Loading models on {DEVICE}...")
    
    # CLIP: Optimized with use_fast
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    
    # BLIP: Base version is lighter for 1GB RAM limits
    bp = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    bm = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE).eval()
    
    gc.collect()
    return cp, cm, bp, bm

clip_processor, clip_model, blip_processor, blip_model = load_ai_models()

# ---------------- UTILS & ANALYSIS ----------------
def generate_caption(frame):
    """Fast captioning with frame resizing to prevent 20s timeout."""
    try:
        # Resize to 224x224 so the AI 'looks' at less data (much faster)
        small_frame = cv2.resize(frame, (224, 224))
        img = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
        
        inputs = blip_processor(img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=25)
        
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        
        # Immediate Memory Cleanup
        del inputs, img, small_frame
        return caption
    except Exception as e:
        return f"Frame analysis error: {e}"

def clip_embedding_batch(frames):
    """Batch process embeddings to save CPU cycles."""
    if not frames or clip_processor is None: return []
    
    # Resize frames for faster embedding generation
    images = [Image.fromarray(cv2.resize(f, (224, 224))) for f in frames]
    inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
    
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    
    return embeddings.cpu().numpy()

# ---------------- SMART VIDEO ANALYSIS ----------------
def analyze_video_resource_friendly(video_path):
    """
    Analyzes the video using Smart Sampling.
    Processes only 1 frame every 2 seconds to avoid the 20s heartbeat bust.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Sample 1 frame every 2 seconds
    sample_rate = int(fps * 2) if fps > 0 else 30
    
    analyzed_data = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_count % sample_rate == 0:
            caption = generate_caption(frame)
            analyzed_data.append(caption)
            # Help the server's RAM
            gc.collect()
            
        frame_count += 1
        
        # Hard limit: Don't analyze more than 30 samples per clip
        if len(analyzed_data) >= 30: break 
        
    cap.release()
    return " ".join(analyzed_data)

# ---------------- GEMINI SELECTION ----------------
def select_clips_with_ai(user_selection_prompt, analyzed_clips, gemini_model):
    """Uses Gemini to filter the catalog."""
    catalog = ""
    for clip in analyzed_clips:
        catalog += f"File: {clip['name']} | Description: {clip['captions']}\n"

    prompt = f"""
    You are a video editor. Return a JSON list of filenames matching: "{user_selection_prompt}"
    
    LIBRARY:
    {catalog}

    Return ONLY a valid JSON list. No text.
    """

    try:
        response = gemini_model.generate_content(prompt)
        clean_json = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        print(f"⚠️ Gemini Error: {e}")
        return [c['name'] for c in analyzed_clips]
