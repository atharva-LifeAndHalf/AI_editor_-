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
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration
)
from config import DEVICE, OUTPUT_DIR, VIDEO_DIR

# --- 1. THE "NO-BUST" FOLDER CHECK ---
# If these folders don't exist, the app busts the second it tries to save.
for folder in [OUTPUT_DIR, VIDEO_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

# --- 2. THE STABILITY PATCHES ---
os.environ["HF_HUB_READ_TIMEOUT"] = "120"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
# This prevents Torch from hogging memory it doesn't need
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- MEDIAPIPE IMPORT ---
try:
    import mediapipe as mp
    try:
        from mediapipe.python.solutions import selfie_segmentation as mp_selfie_seg
    except (ImportError, AttributeError):
        mp_selfie_seg = mp.solutions.selfie_segmentation
    segmentor = mp_selfie_seg.SelfieSegmentation(model_selection=1)
except Exception as e:
    segmentor = None

# ---------------- STYLES ----------------
STYLES = {
    "cinematic": {"scene_threshold": 0.70, "frame_similarity": 0.96, "transition": "crossfade", "transition_frames": 25, "filter": "cinematic"},
    "fast_reel": {"scene_threshold": 0.60, "frame_similarity": 0.93, "transition": "cut", "transition_frames": 0, "filter": "bright"},
    "documentary": {"scene_threshold": 0.80, "frame_similarity": 0.97, "transition": "crossfade", "transition_frames": 15, "filter": "none"},
    "vlog": {"scene_threshold": 0.65, "frame_similarity": 0.94, "transition": "cut", "transition_frames": 5, "filter": "none"}
}

# ---------------- 3. CACHED MODELS (THE RAM SAVER) ----------------
@st.cache_resource
def load_models_safely():
    """Loads models in a sequence that prevents the 1GB RAM spike."""
    print(f"🔹 Initializing AI on {DEVICE}...")
    
    # Load CLIP first, then clear RAM
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    gc.collect()

    # Load BLIP second
    bp = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    bm = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE).eval()
    
    gc.collect()
    return cp, cm, bp, bm

clip_processor, clip_model, blip_processor, blip_model = load_models_safely()

# ---------------- UTILS ----------------
def cosine(a, b):
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0: return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def clip_embedding_batch(frames):
    if not frames or clip_processor is None: return []
    # RESIZE frames to 224x224 - THIS IS CRITICAL TO PREVENT BUSTING
    images = [Image.fromarray(cv2.resize(f, (224, 224))) for f in frames]
    inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings.cpu().numpy()

def generate_caption(frame):
    """The function that was likely timing out at 20 seconds."""
    try:
        if blip_processor is None: return "Video frame"
        # Force tiny frame size for AI analysis
        small_frame = cv2.resize(frame, (224, 224))
        img = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
        
        inputs = blip_processor(img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            # max_new_tokens=20 makes the AI stop talking faster
            out = blip_model.generate(**inputs, max_new_tokens=20)
        
        res = blip_processor.decode(out[0], skip_special_tokens=True)
        
        # MANUALLY PURGE RAM AFTER EVERY CAPTION
        del inputs, img, small_frame
        gc.collect() 
        return res
    except Exception as e:
        return "Analysis error"

# ---------------- GEMINI SELECTION ----------------
def select_clips_with_ai(user_selection_prompt, analyzed_clips, gemini_model):
    catalog = ""
    for clip in analyzed_clips:
        catalog += f"Filename: {clip['name']} | Description: {clip['captions']}\n"

    prompt = f"User wants: {user_selection_prompt}\nClips:\n{catalog}\nReturn ONLY a JSON list of strings (filenames)."

    try:
        response = gemini_model.generate_content(prompt)
        clean_json = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json)
    except Exception as e:
        return [c['name'] for c in analyzed_clips]
