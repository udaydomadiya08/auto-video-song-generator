import os
import base64
from io import BytesIO
from PIL import Image
import requests
from nltk.tokenize import sent_tokenize
from collections import defaultdict

from moviepy.editor import (
    AudioFileClip, ImageClip, CompositeVideoClip,
    concatenate_videoclips, concatenate_audioclips,
    TextClip
)

from moviepy.video.fx.all import resize
from moviepy.video.tools.subtitles import SubtitlesClip

import spacy
import whisperx

from Wav2Lip.inference import parser, run_inference

import os
import time
import random
import json
from datetime import datetime
from pathlib import Path

import os
import json
import random
import time
from datetime import datetime
import google.generativeai as genai
from collections import defaultdict

class GeminiResponse:
    def __init__(self, text):
        self.text = text

FAILED_KEYS_FILE = "disabled_keys.json"
USAGE_FILE = "usage_counts.json"
DAILY_LIMIT = 500
PER_MINUTE_LIMIT = 10

# In-memory per-minute usage tracker
minute_usage_tracker = defaultdict(list)

def load_json_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}

def save_json_file(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def load_disabled_keys():
    data = load_json_file(FAILED_KEYS_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    return set(data.get(today, []))

def save_disabled_key(api_key):
    today = datetime.now().strftime("%Y-%m-%d")
    data = load_json_file(FAILED_KEYS_FILE)

    if today not in data:
        data[today] = []
    if api_key not in data[today]:
        data[today].append(api_key)

    save_json_file(FAILED_KEYS_FILE, data)

def increment_usage(api_key):
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    # Daily count
    usage = load_json_file(USAGE_FILE)
    if today not in usage:
        usage[today] = {}
    if api_key not in usage[today]:
        usage[today][api_key] = 0
    usage[today][api_key] += 1
    save_json_file(USAGE_FILE, usage)

    # Per-minute in memory
    minute = now.strftime("%Y-%m-%d %H:%M")
    minute_usage_tracker[api_key] = [ts for ts in minute_usage_tracker[api_key] if ts.startswith(minute)]
    minute_usage_tracker[api_key].append(now.strftime("%Y-%m-%d %H:%M:%S"))

def has_exceeded_daily_limit(api_key, limit=DAILY_LIMIT):
    today = datetime.now().strftime("%Y-%m-%d")
    usage = load_json_file(USAGE_FILE)
    return usage.get(today, {}).get(api_key, 0) >= limit


def has_exceeded_minute_limit(api_key, limit=PER_MINUTE_LIMIT):
    current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
    recent_times = [ts for ts in minute_usage_tracker[api_key] if ts.startswith(current_minute)]
    return len(recent_times) >= limit


def generate_gemini_response(prompt, model_name=None, max_retries=1111, wait_seconds=5):
    api_keys = [
        "AIzaSyCG-BB-0iP8bTiTJiT9ZgC5eJkzDftV28I",
        "AIzaSyBGpWydub8jBjKW_JM808Q57x_KSVg1Fxw",
        "AIzaSyD-7i3eVHY_tQBlLedDGUYb12tPm88F2bg",
        "AIzaSyCT-678mR3ur4beLyWJJ-QdWA8W8cHvWtM",
        "AIzaSyBnKryfOjV-XsR0tdXWdYv4MXnbvvh_QWU",
        "AIzaSyBenRCth2XXKL6BXh_gRtDAznPfbTd9t4k",
        "AIzaSyCG6iIVxuoPAwRC8FL0DMHhywAFg58vxbM",
        "AIzaSyCWyJeh999WPRt5Mf8hgAfT78hkl_oyy3I",
        "AIzaSyDQoF2-V-jPVinMIHIs4Dts8KPpXeL-5_E",
        "AIzaSyA5VLL-EFpKs2Z0iVdLLK6ir_n9-b1wtrc",
        "AIzaSyCRNFcI51fF1KoS3YbBnaGtFMLIhSqnaSs",
        "AIzaSyCkhmr6hYUKCQMVuNaVMwhUmfLIrvOMn7g",
        "AIzaSyBFZS7DX_wDWvWjln22G3zN2XjORuMJV5o",
        "AIzaSyDOFT9J2OlqyR2KhhMP9qBaE3LqeLQLaIc",
        "AIzaSyDDBqYMNprBSxs006y_Mjm-2iFsedqvyE4",
        "AIzaSyAv7KdGJul7xb5tCnx2bLqZEStXTTtY-NA",
        "AIzaSyAFn_8ws-tj-ix7R_MIvTg-REUZ-93riZo",
        "AIzaSyB9ESuSqJMbEnAdvBKxaGJsfTrkdvaobYc",
        "AIzaSyCqXHtA2dl3tUumG21cMwbhxQdVP9LzypY",
        "AIzaSyAQCRR1KSbgiF3OkXUOInZOntFw1VU4n4k",
        "AIzaSyChdyTOEFX11YlnDaLKMh7IAXA_OzxpWSg",
        "AIzaSyBcBKM39mCY2x2-90tId2LRRQbOzwefLpE",

    ]

    model_names = [
        "gemini-2.5-flash-preview-05-20"
    ]

    disabled_keys_today = load_disabled_keys()

    for attempt in range(max_retries):
        available_keys = [
            k for k in api_keys
            if k not in disabled_keys_today and not has_exceeded_daily_limit(k) and not has_exceeded_minute_limit(k)
        ]

        if not available_keys:
            raise RuntimeError("❌ All API keys are either disabled or have reached daily/minute limits.")

        key = random.choice(available_keys)
        model = model_name or random.choice(model_names)

        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            gemini = genai.GenerativeModel(model)
            print(f"✅ Using model: {model}, key ending in: {key[-6:]}")
            
            response = gemini.generate_content(prompt)
            increment_usage(key)
            return GeminiResponse(response.text.strip())

        except Exception as e:
            print(f"❌ API call failed with key {key[-6:]} (Attempt {attempt + 1}): {e}")
            save_disabled_key(key)
            time.sleep(wait_seconds)

    raise RuntimeError("❌ All Gemini API attempts failed after retries.")




import sys
import os
sys.path.append(os.path.abspath('Wav2Lip'))

def run_wav2lip_inference(
    checkpoint_path,
    face_video,
    audio_path,
    output_video,
    static=True,
    fps=24,
    wav2lip_batch_size=128,   # default value
    resize_factor=1,
    out_height=480
):
    args_list = [
        '--checkpoint_path', checkpoint_path,
        '--face', face_video,
        '--audio', audio_path,
        '--outfile', output_video,
        '--fps', str(fps),
        '--wav2lip_batch_size', str(wav2lip_batch_size),
        '--resize_factor', str(resize_factor),
        '--out_height', str(out_height),
    ]

    if static:
        args_list.append('--static')

    args = parser.parse_args(args_list)

    print("Starting Wav2Lip inference...")
    run_inference(args)
    print("Inference done!")

# === Setup ===
nlp = spacy.load("en_core_web_sm")  # Load spaCy model once

# Gemini API keys (replace with your keys)
api_keys = [
        "AIzaSyCG-BB-0iP8bTiTJiT9ZgC5eJkzDftV28I",
        "AIzaSyBGpWydub8jBjKW_JM808Q57x_KSVg1Fxw",
        "AIzaSyD-7i3eVHY_tQBlLedDGUYb12tPm88F2bg",
        "AIzaSyCT-678mR3ur4beLyWJJ-QdWA8W8cHvWtM",
        "AIzaSyBnKryfOjV-XsR0tdXWdYv4MXnbvvh_QWU",
        "AIzaSyBenRCth2XXKL6BXh_gRtDAznPfbTd9t4k",
        "AIzaSyCG6iIVxuoPAwRC8FL0DMHhywAFg58vxbM",
        "AIzaSyCWyJeh999WPRt5Mf8hgAfT78hkl_oyy3I",
        "AIzaSyDQoF2-V-jPVinMIHIs4Dts8KPpXeL-5_E",
        "AIzaSyA5VLL-EFpKs2Z0iVdLLK6ir_n9-b1wtrc",
        "AIzaSyCRNFcI51fF1KoS3YbBnaGtFMLIhSqnaSs",
        "AIzaSyCkhmr6hYUKCQMVuNaVMwhUmfLIrvOMn7g",
        "AIzaSyBFZS7DX_wDWvWjln22G3zN2XjORuMJV5o",
        "AIzaSyDOFT9J2OlqyR2KhhMP9qBaE3LqeLQLaIc",
        "AIzaSyDDBqYMNprBSxs006y_Mjm-2iFsedqvyE4",
        "AIzaSyAv7KdGJul7xb5tCnx2bLqZEStXTTtY-NA",
        "AIzaSyAFn_8ws-tj-ix7R_MIvTg-REUZ-93riZo",
        "AIzaSyB9ESuSqJMbEnAdvBKxaGJsfTrkdvaobYc",
        "AIzaSyCqXHtA2dl3tUumG21cMwbhxQdVP9LzypY",
        "AIzaSyAQCRR1KSbgiF3OkXUOInZOntFw1VU4n4k",
        "AIzaSyChdyTOEFX11YlnDaLKMh7IAXA_OzxpWSg",
        "AIzaSyBcBKM39mCY2x2-90tId2LRRQbOzwefLpE",

    ]
# Google TTS service account json path
TTS_JSON_KEY_PATH = "striking-shift-463108-p4-add7aacc0dd1.json"

# Video size & font for subtitles
VIDEO_SIZE = (1080, 1920)
FONT_PATH = "/Users/uday/Downloads/VIDEOYT/Anton-Regular.ttf"
FONT_SIZE = 110

# Create needed directories once
os.makedirs("audio", exist_ok=True)
os.makedirs("temp/images", exist_ok=True)
os.makedirs("video_created", exist_ok=True)

# === Gemini Image Generation ===
from google import genai
from google.genai import types

def resize_to_1080x1920_stretch(image: Image.Image) -> Image.Image:
    return image.resize((1080, 1920), Image.LANCZOS)

def compress_image(input_image: Image.Image, output_path: str, max_size_kb=2048):
    quality = 90
    while quality >= 20:
        input_image.save(output_path, format="JPEG", quality=quality)
        if os.path.getsize(output_path) <= max_size_kb * 1024:
            break
        quality -= 5
    return output_path

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import time
import random
import os

def resize_to_1080x1920_stretch(image: Image.Image) -> Image.Image:
    return image.resize((1080, 1920), Image.LANCZOS)

def compress_image(input_image: Image.Image, output_path: str, max_size_kb=2048):
    quality = 90
    while quality >= 20:
        input_image.save(output_path, format="JPEG", quality=quality)
        if os.path.getsize(output_path) <= max_size_kb * 1024:
            break
        quality -= 5
    return output_path

def get_best_song_title_for_yt(original_title):
    prompt = f"""
You are a viral music title expert and YouTube SEO strategist. Your job is to generate one extremely clickable, emotional, and SEO-optimized YouTube title for a new song called: "{original_title}".

🎯 GOAL: The title should grab attention instantly, go viral, rank on YouTube, and emotionally connect with the audience. It must drive massive clicks and views.

📌 RULES:
- Start with or include the song name (or slight remix)
- Must sound like a **viral hit** or **must-listen track**
- Add trending emotional/SEO keywords: "viral", "official", "2025", "heartbreaking", "emotional", "anthem", "sad", "hit", "must hear"
- Keep it short: UNDER 12 words
- Avoid quotes or unnecessary words
- Make it sound powerful, epic, unforgettable

🔁 STYLE EXAMPLES:
- "{original_title} – The Viral Anthem Everyone’s Obsessed With"
- "{original_title} – Official Music Video That Broke the Internet"
- "{original_title} – 2025’s Most Emotional Hit"
- dont use "|" mind it, instaed use "-"
Respond ONLY with the final title. No explanation.
    """

    response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
    return response.text.strip()


def generate_image_for_song(topic: str, max_retries=50, wait_seconds=5) -> str:
    # prompt = (
    #     f"Create a visually rich, vertical image (1080x1920, 9:16) for a music video titled '{topic}'. "
    #     "The image should be immersive and emotionally resonant, using vibrant, high-quality background visuals that reflect the mood of the music. "
    #     "Incorporate abstract elements like light streaks, particles, or dreamy landscapes to hold visual interest throughout the video. "
    #     "Overlay the song title in a minimal, elegant font — small and subtly placed in a bottom corner or with low opacity, "
    #     "so it does not distract from the visual focus but still credits the song. "
    #     "Avoid clutter, use modern cinematic composition, and ensure the image maintains viewer engagement for the full song duration."
    # )
    # prompt = (
    #     f"Create a stunning, scroll-stopping vertical image (1080x1920, 9:16) for a music video titled '{topic.upper()}'. "
    #     "The image must be visually intense, cinematic, and designed to immediately grab attention on mobile screens. "
    #     "Use high-quality, emotionally charged visuals with vibrant colors, surreal or abstract themes, and energetic lighting (e.g., particles, glow, lens flares, dreamy landscapes). "
    #     "The song title should be boldly overlaid in large, ALL CAPS, using a modern sans-serif font with high contrast. "
    #     "Position the text in the upper or center area, making it readable and dominant without obstructing the main visual theme. "
    #     "Ensure the composition feels premium, immersive, and designed for maximum user engagement in short-form content like YouTube Shorts or Instagram Reels. "
    #     "No logos, no watermarks, just powerful imagery and bold text overlay."
    # )
    prompt = (
        f"Create a bright, cheerful, and scroll-stopping vertical image (1080x1920, 9:16) for a kids music video titled '{topic.upper()}'. "
        "The image must be fun, colorful, and designed to instantly attract kids and parents on mobile screens. "
        "Use high-quality, joyful visuals with vibrant colors, playful cartoon-style or animated themes, and soft lighting (e.g., sparkles, rainbows, balloons, clouds, stars). "
        "The song title should be boldly overlaid in large, ALL CAPS, using a bubbly or kid-friendly font with strong contrast. "
        "Position the text in the upper or center area, making it easy to read and eye-catching without hiding key elements of the artwork. "
        "Ensure the design feels wholesome, energetic, and optimized for maximum engagement in short-form kids content like YouTube Shorts or Instagram Reels. "
        "No logos, no watermarks — just pure fun and bold, playful text overlay."
    )



    disabled_keys_today = load_disabled_keys()

    for attempt in range(max_retries):
        available_keys = [
            k for k in api_keys
            if k not in disabled_keys_today
            and not has_exceeded_daily_limit(k, limit=100)
            and not has_exceeded_minute_limit(k)
        ]

        if not available_keys:
            print("❌ All Gemini API keys are either disabled or at daily/minute limit.")
            return None

        key = random.choice(available_keys)
        print(f"🔁 Trying Gemini API key ending with: {key[-6:]}")
        
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
            )
            
            increment_usage(key)

            parts = response.candidates[0].content.parts
            if not parts:
                print("❌ Gemini response missing image data.")
                save_disabled_key(key)
                disabled_keys_today.add(key)
                time.sleep(wait_seconds)
                continue

            for part in parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    data = part.inline_data.data
                    image_data = base64.b64decode(data) if isinstance(data, str) else data

                    if image_data:
                        try:
                            image = Image.open(BytesIO(image_data)).convert("RGB")
                            image = resize_to_1080x1920_stretch(image)
                            os.makedirs("temp/images", exist_ok=True)
                            output_path = f"temp/images/img_{topic}.jpg"
                            compress_image(image, output_path)
                            print(f"✅ Image saved: {output_path}")
                            return output_path
                        except Exception as e:
                            print("❌ Error processing image:", e)
                            return None

            print("❌ No valid image part found in response.")
            save_disabled_key(key)
            disabled_keys_today.add(key)
            time.sleep(wait_seconds)

        except Exception as e:
            print(f"❌ Failed with key {key[-6:]} (Attempt {attempt + 1}): {e}")
            save_disabled_key(key)
            disabled_keys_today.add(key)
            time.sleep(wait_seconds)

    print("❌ All Gemini API attempts failed.")
    return None



import os
from moviepy.editor import *

import os

import os
import re

def clean_song_name(raw_name):
    # Remove content in brackets, parentheses, and after dashes
    cleaned = re.sub(r"[\(\[\{].*?[\)\]\}]", "", raw_name)  # Remove (..), [..], {..}
    cleaned = re.sub(r"[-_].*", "", cleaned)                # Remove after dash/underscore
    cleaned = re.sub(r"\d+", "", cleaned)                   # Remove numbers
    cleaned = re.sub(r"\s+", " ", cleaned)                  # Collapse multiple spaces
    return cleaned.strip().title()                          # Title case

def get_one_song_from_folder(folder_path, extensions=('.mp3', '.wav', '.flac', '.aac')):
    for file in os.listdir(folder_path):
        if file.lower().endswith(extensions):
            raw_name = os.path.splitext(file)[0]
            song_name = clean_song_name(raw_name)
            song_path = os.path.join(folder_path, file)
            return song_name, song_path
    return None, None  # If no song found


from moviepy.editor import *

from send2trash import send2trash
import os

from moviepy.editor import *
import os
import subprocess

from moviepy.editor import *
import os
import subprocess

from pydub import AudioSegment
import os
import subprocess



from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    VideoClip,
    CompositeVideoClip,
    TextClip
)
from moviepy.video.fx.all import fadein, fadeout, resize
import whisperx
import numpy as np
import random
import colorsys
import os

# === Gradient Creation ===
def create_gradient_frame(w, h, offset, direction, left_color, right_color):
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)

    if direction == "diagonal_tl_br":
        pos = (X + Y) / 2
        ratio = (pos + offset) % 1
    elif direction == "diagonal_br_tl":
        pos = (X + Y) / 2
        ratio = (1 - pos + offset) % 1
    else:
        ratio = np.full((h, w), offset)

    gradient = (1 - ratio[..., None]) * left_color + ratio[..., None] * right_color
    return gradient.astype(np.uint8)

def random_bright_color():
    h = random.random()
    s = 1.0
    v = 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return np.array([r, g, b]) * 255

# === Gradient Subtitle Word Clip ===
def create_word_gradient_clip(word, duration, font, fontsize, video_size):
    text_clip = TextClip(word, fontsize=fontsize, font=font, color="white", method="label")
    text_mask = text_clip.to_mask()
    w, h = text_mask.size

    left_color = random_bright_color()
    right_color = random_bright_color()
    direction = random.choice(["diagonal_tl_br", "diagonal_br_tl"])

    def make_frame(t):
        progress = (t / duration) % 1
        offset = progress
        gradient = create_gradient_frame(w, h, offset, direction, left_color, right_color)
        mask_frame = text_mask.get_frame(t)
        colored_frame = (gradient * mask_frame[:, :, None]).astype(np.uint8)
        return colored_frame

    return VideoClip(make_frame, duration=duration).set_position("center").set_mask(text_mask)

# === Subtitle Composer ===
def create_word_by_word_subtitles(
    word_segments,
    video_size=(1080, 1920),
    font="/Users/uday/Downloads/VIDEOYT/Anton-Regular.ttf",
    fontsize=110,
):
    clips = []
    for word_info in word_segments:
        word = word_info["word"]
        start = word_info["start"]
        end = word_info["end"]
        duration = end - start

        grad_clip = create_word_gradient_clip(word, duration, font, fontsize, video_size)
        grad_clip = grad_clip.set_start(start).fx(fadein, 0.05).fx(fadeout, 0.05)
        clips.append(grad_clip)

    return CompositeVideoClip(clips, size=video_size)

# === WhisperX Setup ===

from pydub import AudioSegment

def preprocess_audio(input_path, output_path, boost_db=6):
    audio = AudioSegment.from_file(input_path)
    
    # Optional: normalize first
    normalized = audio.apply_gain(-audio.dBFS)

    # Then boost overall volume
    boosted = normalized + boost_db

    # Export as WAV (lossless) for WhisperX
    boosted.export(output_path, format="wav")
    return output_path


device = "cpu"
print(f"Using device for WhisperX: {device}")
whisper_model = whisperx.load_model("tiny.en", device=device, compute_type="float32")
align_model, align_metadata = whisperx.load_align_model(language_code="en", device=device)

# === Scene Creation (Image + Audio + Gradient Subtitle) ===
def create_scene_clip(sentence, image_path, audio_path, video_size=(1080, 1920)):
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration

    img_clip = ImageClip(image_path).set_duration(duration).resize(video_size).set_fps(30)

    # Transcribe & align audio for subtitles
    print(f"🧠 Transcribing: {os.path.basename(audio_path)}")
    result = whisper_model.transcribe(audio_path)
    aligned_result = whisperx.align(result["segments"], align_model, align_metadata, audio_path, device)
    word_segments = aligned_result["word_segments"]

    # Create animated gradient word-by-word subtitle
    subtitle_clip = create_word_by_word_subtitles(word_segments, video_size=video_size)

    # Compose final scene
    scene = CompositeVideoClip([
        img_clip,
        subtitle_clip.set_duration(duration)
    ])
    scene = scene.set_audio(audio_clip).set_duration(duration)

    return scene



import os
import subprocess
from moviepy.editor import (
    AudioFileClip, ImageClip, VideoFileClip, CompositeVideoClip, vfx
)
from pydub import AudioSegment

def create_video_from_song_and_image(song_path, image_path, output_path, particle_overlay_path=None, overlay_opacity=0.15):
    try:
        audio_clip = AudioFileClip(song_path)
        duration = audio_clip.duration
        print(f"🎧 Audio duration: {duration:.2f} sec")
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return

    # 1. Base image clip
    image_clip = ImageClip(image_path).set_duration(duration).set_fps(30).resize((1080, 1920))

    # # 2. Particle overlay (optional)
    final_clip = image_clip
    # if particle_overlay_path and os.path.exists(particle_overlay_path):
    #     try:
    #         particle_clip = VideoFileClip(particle_overlay_path, audio=False).resize((1080, 1920)).set_opacity(overlay_opacity).set_fps(30)
    #         particle_clip = particle_clip.fx(vfx.loop, duration=duration)
    #         final_clip = CompositeVideoClip([image_clip, particle_clip])
    #     except Exception as e:
    #         print(f"❌ Particle overlay error: {e}")

    # 3. Generate animated gradient subtitles
    # # 3. Generate animated gradient subtitles
    # try:
    #     preprocessed_path = preprocess_audio(song_path, "temp/boosted.wav")

    #     result = whisper_model.transcribe(preprocessed_path)
    #     aligned_result = whisperx.align(result["segments"], align_model, align_metadata, preprocessed_path, device)
    #     word_segments = aligned_result.get("word_segments", [])

    #     if not word_segments:
    #         print(f"⚠️ No speech detected in: {preprocessed_path}. Skipping subtitles.")
    #     else:
    #         subtitle_clip = create_word_by_word_subtitles(
    #             word_segments,
    #             video_size=(1080, 1920),
    #             font="/Users/uday/Downloads/VIDEOYT/Anton-Regular.ttf",
    #             fontsize=110,
    #         ).set_duration(duration)
    #         final_clip = CompositeVideoClip([final_clip, subtitle_clip])
    # except Exception as e:
    #     print(f"❌ Subtitle generation failed: {e}")

    # 4. Export video without audio (temporary file)
    video_no_audio = output_path.replace(".mp4", "_noaudio.mp4")
    try:
        final_clip.write_videofile(
            video_no_audio,
            codec="libx264",
            preset="ultrafast",
            threads=os.cpu_count(),
            audio=False,
            verbose=False
        )
    except Exception as e:
        print(f"❌ Video export failed: {e}")
        return

    # 5. Convert audio to WAV for ffmpeg
    wav_path = song_path.replace(".mp3", "_converted.wav")
    try:
        audio = AudioSegment.from_file(song_path)
        audio.export(wav_path, format="wav")
    except Exception as e:
        print(f"❌ Audio conversion to WAV failed: {e}")
        return

    # 6. Mux final audio + video
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_no_audio,
            "-i", wav_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ Final video created with audio: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg muxing failed: {e}")
        return

    # 7. Cleanup
    os.remove(video_no_audio)
    os.remove(wav_path)

from datetime import datetime

def process_all_songs():
    folder_path = "songs"
    output_dir = "songer"

    while True:
        song_name, song_path = get_one_song_from_folder(folder_path)
        song_name = get_best_song_title_for_yt(song_name)
        if not song_path:
            print("✅ All songs processed. Folder is empty.")
            break

        print(f"\n🎵 Processing: {song_name}")

        # Generate image
        image_path = generate_image_for_song(song_name)
        if not image_path:
            print(f"❌ Failed to generate image for: {song_name}")
            continue

        # Output filename with datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{song_name}_{timestamp}.mp4"
        output_video_path = os.path.join(output_dir, output_filename)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Create video
            create_video_from_song_and_image(song_path, image_path, output_video_path, particle_overlay_path="05734_huge_dust_particles_overlay_wwwcutestockfootagecom (1).mp4")

            def save_file_topic_mappings(mappings, filename):
                """
                Save mappings of output file paths and topic names to a text file.
                
                Each line will be:
                output_file_path | topic_name
                
                :param mappings: List of tuples [(file_path1, topic1), (file_path2, topic2), ...]
                :param filename: Text file to save mappings
                """
                with open(filename, 'a') as f:
                    for file_path, topic in mappings:
                        f.write(f"{file_path} | {topic}\n")

            # Example usage:
            mappings = [
                (output_video_path, song_name)
                
            ]

            save_file_topic_mappings(mappings, "/Users/uday/Downloads/VIDEOYT/file_topic_map_song.txt")
            # Delete song file
            send2trash(song_path)
            print(f"🗑️ Deleted song: {song_path}")

            # Delete image file
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"🗑️ Deleted image: {image_path}")
        except Exception as e:
            print(f"❌ Error creating video for {song_name}: {e}")



process_all_songs()

