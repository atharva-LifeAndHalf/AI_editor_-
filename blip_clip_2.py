import os
import cv2
import gc
import torch
import json
import numpy as np
import streamlit as st # Added for caching stability
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration
)
from config import DEVICE, OUTPUT_DIR, VIDEO_DIR

# --- 2025 STABILITY PATCHES ---
os.environ["HF_HUB_READ_TIMEOUT"] = "120"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

# --- MEDIAPIPE IMPORT ---
try:
    import mediapipe as mp
    try:
        from mediapipe.python.solutions import selfie_segmentation as mp_selfie_seg
    except (ImportError, AttributeError):
        mp_selfie_seg = mp.solutions.selfie_segmentation
    segmentor = mp_selfie_seg.SelfieSegmentation(model_selection=1)
    print(" MediaPipe initialized")
except Exception as e:
    print(f"MediaPipe unavailable: {e}")
    segmentor = None

# ---------------- STYLES ----------------
STYLES = {
    "cinematic": {"scene_threshold": 0.70, "frame_similarity": 0.96, "transition": "crossfade", "transition_frames": 25,
                  "filter": "cinematic"},
    "fast_reel": {"scene_threshold": 0.60, "frame_similarity": 0.93, "transition": "cut", "transition_frames": 0,
                  "filter": "bright"},
    "documentary": {"scene_threshold": 0.80, "frame_similarity": 0.97, "transition": "crossfade",
                    "transition_frames": 15, "filter": "none"},
    "product": {"scene_threshold": 0.75, "frame_similarity": 0.96, "transition": "dissolve", "transition_frames": 30,
                "filter": "bright"},
    "horror": {"scene_threshold": 0.75, "frame_similarity": 0.97, "transition": "fade_to_black",
               "transition_frames": 20, "filter": "dark"},
    "vlog": {"scene_threshold": 0.65, "frame_similarity": 0.94, "transition": "cut", "transition_frames": 5,
             "filter": "none"},
    "vintage": {"scene_threshold": 0.70, "frame_similarity": 0.95, "transition": "dissolve", "transition_frames": 20,
                "filter": "vintage"},
    "music_video": {"scene_threshold": 0.65, "frame_similarity": 0.94, "transition": "crossfade",
                    "transition_frames": 15, "filter": "contrast"}
}

# ---------------- LOAD MODELS (CACHED) ----------------
@st.cache_resource
def load_models_cached():
    print(f"🔹 Loading AI models on {DEVICE}...")
    try:
        cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
        print(" CLIP loaded")
    except Exception as e:
        print(f" CLIP error: {e}")
        cp, cm = None, None

    try:
        bp = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        bm = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE).eval()
        print(" BLIP loaded")
    except Exception as e:
        print(f" BLIP error: {e}")
        bp, bm = None, None
    
    gc.collect()
    return cp, cm, bp, bm

# Initialize from cache
clip_processor, clip_model, blip_processor, blip_model = load_models_cached()

# ---------------- UTILS ----------------
def cosine(a, b):
    """Essential for processing_1.py semantic ordering"""
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def clip_embedding_batch(frames):
    """Batch process CLIP embeddings with resizing for speed"""
    if not frames or clip_processor is None or clip_model is None:
        return []
    # Resize frames to 224x224 to prevent memory bust
    images = [Image.fromarray(cv2.resize(f, (224, 224))) for f in frames]
    inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings.cpu().numpy()

def generate_caption(frame):
    """Generate caption using BLIP with 224px resizing to stay under 20s"""
    try:
        if blip_processor is None or blip_model is None:
            return "Video frame"
        # Resize for speed
        small_frame = cv2.resize(frame, (224, 224))
        img = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
        inputs = blip_processor(img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=25)
        
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        del inputs, img, small_frame # Free RAM immediately
        return caption
    except Exception as e:
        print(f" Caption error: {e}")
        return "Video frame"

# ---------------- GEMINI SMART SELECTION ----------------
def select_clips_with_ai(user_selection_prompt, analyzed_clips, gemini_model):
    """Filters the analyzed clips using Gemini based on the BLIP captions."""
    clip_catalog = ""
    for clip in analyzed_clips:
        clip_catalog += f"Filename: {clip['name']} | Description: {clip['captions']}\n"

    prompt = f"""
    You are an expert video editor. I have a library of video clips with descriptions.
    Based on the user's request, return a JSON list of filenames that best match.

    USER REQUEST: "{user_selection_prompt}"

    CLIP LIBRARY:
    {clip_catalog}

    INSTRUCTIONS:
    - Return ONLY a valid JSON list of strings (filenames).
    - If no clips match perfectly, return the filenames of the 3 most relevant clips.
    - Do not include any conversational text, only the JSON.
    """

    try:
        response = gemini_model.generate_content(prompt)
        clean_json = response.text.strip().replace("```json", "").replace("```", "")
        selected_files = json.loads(clean_json)
        return selected_files
    except Exception as e:
        print(f"Gemini Selection Error: {e}")
        return [c['name'] for c in analyzed_clips]
