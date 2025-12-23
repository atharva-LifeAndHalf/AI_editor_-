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
os.environ["HF_HUB_READ_TIMEOUT"] = "120"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

# --- MEDIAPIPE INITIALIZATION ---
try:
    import mediapipe as mp
    from mediapipe.python.solutions import selfie_segmentation as mp_selfie_seg
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

# ---------------- UTILS (FIXED IMPORT ERROR) ----------------
def cosine(a, b):
    """Calculates cosine similarity between two vectors. 
    Required by processing_1.py to order clips."""
    # Ensure inputs are numpy arrays
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------- LOAD MODELS (CACHED) ----------------
@st.cache_resource
def load_ai_models():
    """Loads vision models into memory once."""
    print(f"🔹 Loading models on {DEVICE}...")
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    bp = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    bm = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE).eval()
    gc.collect()
    return cp, cm, bp, bm

clip_processor, clip_model, blip_processor, blip_model = load_ai_models()

# ---------------- ANALYSIS FUNCTIONS ----------------
def generate_caption(frame):
    """Fast captioning with frame resizing."""
    try:
        small_frame = cv2.resize(frame, (224, 224))
        img = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
        inputs = blip_processor(img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=25)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        del inputs, img, small_frame
        return caption
    except Exception as e:
        return "Frame analysis error"

def clip_embedding_batch(frames):
    """Batch process embeddings for CLIP."""
    if not frames or clip_processor is None: return []
    images = [Image.fromarray(cv2.resize(f, (224, 224))) for f in frames]
    inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings.cpu().numpy()

def select_clips_with_ai(user_selection_prompt, analyzed_clips, gemini_model):
    """Uses Gemini to filter the catalog based on descriptions."""
    catalog = ""
    for clip in analyzed_clips:
        catalog += f"File: {clip['name']} | Description: {clip['captions']}\n"

    prompt = f"""
    Return a JSON list of filenames matching: "{user_selection_prompt}"
    LIBRARY:
    {catalog}
    Return ONLY a valid JSON list of strings.
    """
    try:
        response = gemini_model.generate_content(prompt)
        clean_json = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        print(f"⚠️ Gemini Error: {e}")
        return [c['name'] for c in analyzed_clips]
