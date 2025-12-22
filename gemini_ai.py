from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

load_dotenv()
API_KEY = os.getenv("gemini_key")

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        print(" Gemini API configured")
    except Exception as e:
        print(f" Gemini API error: {e}")
        API_KEY = None
else:
    print("Gemini API key not found in .env")

@retry(wait=wait_exponential(multiplier=2, min=5, max=60), stop=stop_after_attempt(3))
def get_gemini_instructions(prompt, video_context):
    """Get AI editing instructions from Gemini"""
    try:
        if not API_KEY:
            return get_default_edits()

        model = genai.GenerativeModel("gemini-2.5-flash")

        context_str = json.dumps(video_context, indent=2)

        sys_msg = (
            f"Video Context:\n{context_str}\n\n"
            f"User Request: {prompt}\n\n"
            "Generate editing instructions as JSON with these exact fields:\n"
            "{\n"
            '  "speed": float (0.5-2.0),\n'
            '  "filter": "cinematic"|"bright"|"vintage"|"dark"|"contrast"|"none",\n'
            '  "stabilize": boolean,\n'
            '  "remove_bg": boolean,\n'
            '  "erase_object": boolean,\n'
            '  "add_logo": boolean,\n'
            '  "trim_start": int (seconds),\n'
            '  "trim_end": int (seconds)\n'
            "}\n\n"
            "Return ONLY the JSON object, no markdown or explanations."
        )

        response = model.generate_content(sys_msg)
        text = response.text.strip().replace("```json", "").replace("```", "").strip()

        edits = json.loads(text)

        # Validate
        edits["speed"] = max(0.5, min(2.0, float(edits.get("speed", 1.0))))
        valid_filters = ["cinematic", "bright", "vintage", "dark", "contrast", "none"]
        if edits.get("filter") not in valid_filters:
            edits["filter"] = "none"

        return edits
    except Exception as e:
        print(f" Gemini error: {e}")
        return get_default_edits()


def get_default_edits():
    return {
        "speed": 1.0,
        "filter": "none",
        "stabilize": False,
        "remove_bg": False,
        "erase_object": False,
        "add_logo": False,
        "trim_start": 0,
        "trim_end": 0
    }

