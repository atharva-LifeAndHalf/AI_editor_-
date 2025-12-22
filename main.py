import os
import json
import cv2
from config import VIDEO_DIR, ANALYSIS_JSON, TEMP_DIR, FINAL_VIDEO, FINAL_FPS, OUTPUT_DIR
import gc
import torch
from processing import semantic_order, process_clip_to_file, combine_videos_with_transitions, apply_ai_edits
from clip_blip import STYLES
from video_analysis import extract_and_analyze_video
from gemini_ai import API_KEY, get_gemini_instructions

def main():

    print("AI video editor")

    # Check video directory
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR, exist_ok=True)
        print(f"\n No videos found. Please add videos to: {VIDEO_DIR}")
        return

    # Find videos
    videos = sorted([f for f in os.listdir(VIDEO_DIR)
                     if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))])

    if not videos:
        print(f"\n No videos found in {VIDEO_DIR}")
        return

    print(f"\nFound {len(videos)} video(s)")

    # Select style
    print(f"\n Available styles: {', '.join(STYLES.keys())}")
    style_name = input("Choose style (default: cinematic): ").strip().lower() or "cinematic"

    if style_name not in STYLES:
        style_name = "cinematic"

    style = STYLES[style_name]
    print(f" Using style: {style_name}")

    # Semantic ordering
    use_semantic = input("Use smart clip ordering? (y/n, default=y): ").strip().lower() != 'n'

    # Phase 1: Analyze all videos
    print("\n" + "=" * 70)
    print(" PHASE 1: VIDEO ANALYSIS")
    print("=" * 70)

    clips = []
    all_analysis = []

    for i, video_name in enumerate(videos):
        print(f"\n▶ Analyzing {i + 1}/{len(videos)}: {video_name}")
        video_path = os.path.join(VIDEO_DIR, video_name)

        analysis = extract_and_analyze_video(video_path)

        if not analysis["frames"]:
            print("    No frames extracted, skipping")
            continue

        print(f"   Duration: {analysis['duration']:.2f}s")
        print(f"   Frames: {analysis['total_frames']}")
        print(f"   Captions: {len(analysis['captions'])}")
        for cap in analysis['captions']:
            print(f"      - {cap}")

        clips.append({
            "name": video_name,
            "path": video_path,
            "frames": analysis["frames"],
            "embedding": analysis["embedding"],
            "duration": analysis["duration"],
            "captions": analysis["captions"]
        })

        all_analysis.append({
            "file": video_name,
            "duration": analysis["duration"],
            "total_frames": analysis["total_frames"],
            "captions": analysis["captions"]
        })

        gc.collect()

    if not clips:
        print("\n No clips processed")
        return

    # Save analysis
    with open(ANALYSIS_JSON, 'w') as f:
        json.dump(all_analysis, f, indent=2)
    print(f"\n Analysis saved to: {ANALYSIS_JSON}")

    # Semantic ordering
    if use_semantic:
        print("\n Ordering clips semantically...")
        clips = semantic_order(clips)
        print("   Order:", " → ".join([c["name"] for c in clips]))

    # Phase 2: Process clips
    print("\n" + "=" * 70)
    print("🎬 PHASE 2: PROCESSING CLIPS")
    print("=" * 70)

    temp_files = []

    for i, clip in enumerate(clips):
        print(f"\n▶ Processing {i + 1}/{len(clips)}: {clip['name']}")
        temp_path = os.path.join(TEMP_DIR, f"temp_{i:03d}.mp4")

        process_clip_to_file(clip, style, temp_path)
        temp_files.append(temp_path)

        del clip["frames"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Phase 3: Combine
    print("\n" + "=" * 70)
    print("🎬 PHASE 3: COMBINING VIDEOS")
    print("=" * 70)

    combine_videos_with_transitions(temp_files, style, FINAL_VIDEO)

    # Get stats
    cap = cv2.VideoCapture(FINAL_VIDEO)
    final_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print("\n" + "=" * 70)
    print("BASE VIDEO COMPLETE")
    print("=" * 70)
    print(f"Output: {FINAL_VIDEO}")
    print(f"Duration: {final_frames / FINAL_FPS:.2f}s")
    print(f"Frames: {final_frames}")

    # Phase 4: AI Editing
    if API_KEY:
        print("\n" + "=" * 70)
        print(" PHASE 4: AI-POWERED EDITING (OPTIONAL)")
        print("=" * 70)

        choice = input("\nApply AI-powered edits? (y/n): ").strip().lower()

        if choice == 'y':
            # Show available options
            print("\n AI Editing Options:")
            print("   - Speed adjustment (0.5x - 2.0x)")
            print("   - Color filters (cinematic, bright, vintage, dark, contrast)")
            print("   - Video stabilization")
            print("   - Background removal")
            print("   - Object removal/inpainting")
            print("   - Logo overlay")
            print("   - Smart trimming")

            prompt = input("\n Describe your edits (e.g., 'speed up to 1.5x, add cinematic filter, stabilize'): ")

            # Prepare context for Gemini
            video_context = {
                "total_clips": len(all_analysis),
                "total_duration": sum(a["duration"] for a in all_analysis),
                "clips": all_analysis
            }

            print("\n Getting AI recommendations...")
            edits = get_gemini_instructions(prompt, video_context)

            print("\n AI Editing Plan:")
            print(json.dumps(edits, indent=2))

            confirm = input("\nApply these edits? (y/n): ").strip().lower()

            if confirm == 'y':
                # Handle object removal ROI selection
                roi = None
                if edits.get("erase_object"):
                    print("\nSelect object to remove:")
                    print("   1. Draw rectangle around object")
                    print("   2. Press SPACE or ENTER")
                    print("   3. Press ESC")

                    cap_roi = cv2.VideoCapture(FINAL_VIDEO)
                    ret, frame = cap_roi.read()

                    if ret:
                        roi = cv2.selectROI("Select Object to Erase", frame, False)
                        cv2.destroyAllWindows()
                        print(f"   ✓ Selected ROI: {roi}")

                    cap_roi.release()

                # Apply edits
                ai_output = FINAL_VIDEO.replace(".mp4", "_ai_edited.mp4")
                apply_ai_edits(FINAL_VIDEO, edits, ai_output, roi)

                # Get final stats
                cap = cv2.VideoCapture(ai_output)
                ai_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ai_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()

                print("\n" + "=" * 70)
                print(" AI-EDITED VIDEO COMPLETE")
                print("=" * 70)
                print(f"Output: {ai_output}")
                print(f"Duration: {ai_frames / ai_fps:.2f}s")
                print(f"Frames: {ai_frames}")
                print(f"Speed: {edits['speed']}x")
                print(f"Filter: {edits['filter']}")
            else:
                print("\n AI edits cancelled")
        else:
            print("\nSkipping AI editing")
    else:
        print("\n Gemini API not configured - AI features unavailable")
        print("   Add 'gemini_key' to .env file to enable AI features")

    # Cleanup
    print("\n Cleaning up temporary files...")
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except:
            pass

    # Final summary
    print("\n" + "=" * 70)
    print(" ALL PROCESSING COMPLETE!")
    print("=" * 70)
    print(f" Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f" Analysis saved: {ANALYSIS_JSON}")
    print(f" Base video: {FINAL_VIDEO}")

    if API_KEY and choice == 'y' and confirm == 'y':
        print(f" AI-edited: {ai_output}")

    file_size = os.path.getsize(FINAL_VIDEO) / (1024 * 1024)
    print(f" File size: {file_size:.2f} MB")
    print("\n Done!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n️ Process interrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback

        traceback.print_exc()