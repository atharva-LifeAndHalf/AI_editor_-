import cv2
from config import ANALYSIS_SAMPLE_RATE, BATCH_SIZE
import numpy as np
from blip_clip_2 import generate_caption, clip_embedding_batch, cosine


def extract_and_analyze_video(path, sample_rate=ANALYSIS_SAMPLE_RATE):
    """Extract frames and generate analysis"""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    frame_idx = 0

    while True:
        ret, f = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        frame_idx += 1

    cap.release()

    # Generate captions for first, middle, last frames
    captions = []
    if frames:
        indices = [0, len(frames) // 2, -1]
        for idx in indices:
            captions.append(generate_caption(frames[idx]))

    # Get CLIP embedding for semantic ordering
    embedding = None
    if frames:
        sample_frames = frames[::max(1, len(frames) // 5)][:5]
        embeddings = clip_embedding_batch(sample_frames)
        if len(embeddings) > 0:
            embedding = np.mean(embeddings, axis=0).tolist()

    return {
        "frames": frames,
        "fps": fps,
        "total_frames": total_frames,
        "duration": total_frames / fps if fps > 0 else 0,
        "captions": captions,
        "embedding": embedding
    }


def segment_scenes(frames, threshold, batch_size=BATCH_SIZE):
    """Scene segmentation with batch processing"""
    if not frames:
        return [], []

    all_embeddings = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        batch_embeddings = clip_embedding_batch(batch)
        all_embeddings.extend(batch_embeddings)

    scenes = []
    current_scene = []
    last_emb = None

    for idx, (frame, emb) in enumerate(zip(frames, all_embeddings)):
        if last_emb is None or cosine(last_emb, emb) > threshold:
            current_scene.append((frame, idx))
        else:
            if current_scene:
                scenes.append(current_scene)
            current_scene = [(frame, idx)]
        last_emb = emb

    if current_scene:
        scenes.append(current_scene)

    return scenes, all_embeddings


def deduplicate_frames(scene_frames, all_embeddings, sim_threshold):
    """Remove duplicate frames"""
    if not scene_frames:
        return []

    selected = []
    last_emb = None

    for frame, idx in scene_frames:
        emb = all_embeddings[idx]
        if last_emb is None or cosine(last_emb, emb) < sim_threshold:
            selected.append(frame)
            last_emb = emb

    if not selected:
        selected = [scene_frames[len(scene_frames) // 2][0]]


    return selected
