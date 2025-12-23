import os
import cv2
import gc
import torch
import json
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration
)
from config import DEVICE, OUTPUT_DIR, VIDEO_DIR

# --- 2025 STABILITY PATCHES: PREVENT BUSTED LINK ---
# We force the system to wait longer for HuggingFace models to download.
# This stops the 'ReadTimeout' error that kills your link.
os.environ["HF_HUB_READ_TIMEOUT"] = "60"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

# --- MEDIAPIPE IMPORT ---
try:
    import mediapipe as mp
    try:
        from mediapipe.python.solutions import selfie_segmentation as mp_selfie_seg
    except (ImportError, AttributeError):
        mp_selfie_seg = mp.solutions.selfie_segmentation
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
    "product": {"scene_threshold": 0.75, "frame_similarity": 0.96, "transition": "dissolve", "transition_frames": 30, "filter": "bright"},
    "horror": {"scene_threshold": 0.75, "frame_similarity": 0.97, "transition": "fade_to_black", "transition_frames": 20, "filter": "dark"},
    "vlog": {"scene_threshold": 0.65, "frame_similarity": 0.94, "transition": "cut", "transition_frames": 5, "filter": "none"},
    "vintage": {"scene_threshold": 0.70, "frame_similarity": 0.95, "transition": "dissolve", "transition_frames": 20, "filter": "vintage"},
    "music_video": {"scene_threshold": 0.65, "frame_similarity": 0.94, "transition": "crossfade", "transition_frames": 15, "filter": "contrast"}
}

# ---------------- LOAD MODELS ----------------
print(f"🔹 Loading AI models on {DEVICE}...")

# CHANGE: Added use_fast=True to stop the "Slow Processor" warning/crash
try:
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    print(" ✅ CLIP loaded")
except Exception as e:
    print(f" ❌ CLIP error: {e}")
    clip_processor, clip_model = None, None

try:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE).eval()
    print(" ✅ BLIP loaded")
except Exception as e:
    print(f" ❌ BLIP error: {e}")
    blip_processor, blip_model = None, None


# ---------------- UTILS ----------------
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def clip_embedding_batch(frames):
    """Batch process CLIP embeddings"""
    if not frames or clip_processor is None or clip_model is None:
        return []
    images = [Image.fromarray(f) for f in frames]
    # CHANGE: Added return_tensors and padding to ensure stability
    inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings.cpu().numpy()

def generate_caption(frame):
    """Generate caption using BLIP"""
    try:
        if blip_processor is None or blip_model is None:
            return "Video frame"
        # BGR to RGB conversion is critical for BLIP accuracy
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = blip_processor(img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = blip_model.generate(**inputs)
        return blip_processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print(f" Caption error: {e}")
        return "Video frame"


# ---------------- GEMINI SMART SELECTION ----------------
def select_clips_with_ai(user_selection_prompt, analyzed_clips, gemini_model):
    """
    Filters the analyzed clips using Gemini based on the BLIP captions.
    """
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
    - No markdown formatting, just the raw JSON list.
    - If no clips match perfectly, return the 3 most relevant.
    """

    try:
        response = gemini_model.generate_content(prompt)
        # Robust cleaning of markdown tags
        clean_json = response.text.strip().replace("```json", "").replace("```", "").strip()
        selected_files = json.loads(clean_json)
        return selected_files
    except Exception as e:
        print(f"⚠️ Gemini Selection Error: {e}. Returning all clips as fallback.")
        return [c['name'] for c in analyzed_clips]
