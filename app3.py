import streamlit as st
import os
import json
import gc
import shutil
import time
from config import VIDEO_DIR, ANALYSIS_JSON, TEMP_DIR, FINAL_VIDEO, OUTPUT_DIR
from processing_1 import semantic_order, process_clip_to_file, combine_videos_with_transitions, apply_ai_edits
from blip_clip_2 import STYLES, select_clips_with_ai
from video_analysis import extract_and_analyze_video
from gemini_ai import API_KEY, get_gemini_instructions as gemini_model

# --- PAGE CONFIG ---
st.set_page_config(page_title="L&H AI Studio", layout="wide")

# --- AUTO-RESET LOGIC ---
# This ensures that every time the link is opened (new session), the data is wiped.
if 'initialized' not in st.session_state:
    for folder in [VIDEO_DIR, TEMP_DIR, OUTPUT_DIR]:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception:
                    pass
    if os.path.exists(ANALYSIS_JSON): os.remove(ANALYSIS_JSON)
    if os.path.exists(FINAL_VIDEO): os.remove(FINAL_VIDEO)
    
    # Mark session as initialized so it doesn't loop reset on every click
    st.session_state['initialized'] = True

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stButton > button:first-child {
        background-color: #6366f1;
        color: white;
        border-radius: 8px;
        padding: 0.6rem;
    }
    .video-card {
        border: 2px solid #334155;
        border-radius: 12px;
        padding: 10px;
        background: #000;
        margin-bottom: 20px;
    }
    video {
        max-width: 100%;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

def get_video_binary(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

def main():
    if os.path.exists("l&h_logo.png"):
        st.sidebar.image("l&h_logo.png")

    st.title("🎥 AI Video Editor")

    with st.sidebar:
        st.header("1. Project Assets")
        uploaded_files = st.file_uploader("Upload internal system clips",
                                          type=['mp4', 'mov', 'avi'],
                                          accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                path = os.path.join(VIDEO_DIR, uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"Internal System updated: {len(uploaded_files)} clips.")

        st.divider()
        st.header("2. Base Settings")
        style_name = st.selectbox("Visual Style", options=list(STYLES.keys()), index=0)
        use_semantic = st.toggle("Smart Clip Ordering", value=True)

        st.divider()
        st.header("3. AI Selection Filter")
        enable_ai_selection = st.toggle("Enable AI Smart Selection", value=False)
        selection_query = ""
        if enable_ai_selection:
            selection_query = st.text_input("Selection Prompt", placeholder="e.g. 'Only urban clips'")

        if st.button("Generate Base Video", use_container_width=True):
            run_generation_flow(style_name, use_semantic, enable_ai_selection, selection_query)

        # MANUAL RESET BUTTON REMOVED AS PER INSTRUCTIONS

    if os.path.exists(FINAL_VIDEO):
        col_preview, col_panel = st.columns([2, 1])

        with col_preview:
            st.subheader("Preview Draft")
            video_data = get_video_binary(FINAL_VIDEO)
            if video_data:
                v_spacer_l, v_main, v_spacer_r = st.columns([0.3, 0.4, 0.3])
                with v_main:
                    st.markdown('<div class="video-card">', unsafe_allow_html=True)
                    st.video(video_data, format="video/mp4")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Preview data not found.")

        with col_panel:
            st.subheader("AI Edits")
            user_prompt = st.text_area("Describe changes:", placeholder="e.g. 'Make it faster'", height=180)
            if st.button("Apply AI Edits", use_container_width=True):
                if API_KEY:
                    run_ai_flow(user_prompt)
                else:
                    st.error("API Key missing.")
            st.divider()
            st.subheader("System Export")
            if video_data:
                st.download_button(label="💾 Download Result", data=video_data, file_name="ai_output.mp4",
                                   mime="video/mp4", use_container_width=True)
    else:
        st.info("System Ready. Please upload clips and generate the base video to begin.")

# --- FLOW LOGIC ---

def run_generation_flow(style_name, use_semantic, enable_ai_selection, selection_query):
    with st.status("🎬 Processing internal files...", expanded=True) as status:
        videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".mov", ".avi"))])
        if not videos:
            st.error("No videos found.")
            return

        all_clips = []
        for video_name in videos:
            video_path = os.path.join(VIDEO_DIR, video_name)
            st.write(f"Analyzing {video_name}...")
            analysis = extract_and_analyze_video(video_path)
            if analysis["frames"]:
                all_clips.append({
                    "name": video_name,
                    "path": video_path,
                    "frames": analysis["frames"],
                    "embedding": analysis["embedding"],
                    "duration": analysis["duration"],
                    "captions": analysis["captions"]
                })
            gc.collect()

        if enable_ai_selection and selection_query:
            st.write("Gemini filtering clips based on prompt...")
            selected_filenames = select_clips_with_ai(selection_query, all_clips, gemini_model)
            final_clips = [c for c in all_clips if c['name'] in selected_filenames]
            st.write(f"🎞️AI selected {len(final_clips)} clips.")
        else:
            final_clips = all_clips

        if use_semantic:
            final_clips = semantic_order(final_clips)

        temp_files = []
        for i, clip in enumerate(final_clips):
            temp_path = os.path.join(TEMP_DIR, f"temp_{i:03d}.mp4")
            process_clip_to_file(clip, STYLES[style_name], temp_path)
            temp_files.append(temp_path)
            del clip["frames"]
            gc.collect()

        combine_videos_with_transitions(temp_files, STYLES[style_name], FINAL_VIDEO)
        status.update(label="Internal View Ready!", state="complete")
        time.sleep(1)
        st.rerun()

def run_ai_flow(prompt):
    with st.spinner("Applying AI instructions..."):
        all_analysis = []
        if os.path.exists(ANALYSIS_JSON):
            with open(ANALYSIS_JSON, 'r') as f:
                all_analysis = json.load(f)
        edits = gemini_model(prompt, {"clips": all_analysis})
        ai_output = os.path.join(OUTPUT_DIR, "ai_final_edit.mp4")
        apply_ai_edits(FINAL_VIDEO, edits, ai_output, None)
        if os.path.exists(ai_output):
            shutil.move(ai_output, FINAL_VIDEO)
            st.rerun()

if __name__ == "__main__":
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
