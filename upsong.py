import os
import socket
socket.setdefaulttimeout(600000)
from collections import defaultdict

# === Imports === #
import os
import requests
import nltk
import spacy
from gtts import gTTS
from datetime import datetime
from nltk.tokenize import sent_tokenize
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, ColorClip
import google.generativeai as genai
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
import ffmpeg
import os
import subprocess
from pydub import AudioSegment
import time
from functools import wraps
import google.generativeai as genai
import json
import os
import time
from datetime import datetime
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
import os

from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

from collections import Counter
import numpy as np

import time

import subprocess
import os
import subprocess
from pydub import AudioSegment
import os
from googleapiclient.http import MediaFileUpload
import json

from googleapiclient.errors import HttpError
import pickle
from googleapiclient.discovery import build
import os
import shutil
from pathlib import Path
import re  
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import google.generativeai as genai
from serpapi import GoogleSearch

import sys
import os
sys.path.append(os.path.abspath('Wav2Lip'))

# Set environment variables so pydub can find ffmpeg and ffprobe

os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

# Optional: also explicitly set converter paths (may be ignored by some methods)
AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"
AudioSegment.ffprobe = "/opt/homebrew/bin/ffprobe"

# CLIENT_SECRETS_FILE = "/Users/uday/Downloads/VIDEOYT/client_secret_.json"  # Path to your client secret
# SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
# API_SERVICE_NAME = "youtube"
# API_VERSION = "v3"

# === Initial Setup === #
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")

# === API KEYS === #
PEXELS_API_KEY = "DGhCtAB83klpCIv5yq5kMIb2zun7q67IvHJysvW4lInb0WVXaQF2xLMu"
SERP_API_KEY = "7f55bbfeff700d39fe9ee306af78102a69cf43267987037a77c5b111cbc48e98"

import os
import time
import random
import json
from datetime import datetime
from pathlib import Path

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



from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import os

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

# ✅ Resize to 1080x1920 (stretched)
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import os
import time
import random

def resize_to_1080x1920_stretch(image: Image.Image) -> Image.Image:
    return image.resize((1080, 1920), Image.LANCZOS)

def compress_thumbnail(input_image: Image.Image, output_path: str, max_size_kb=1024):
    quality = 90
    while quality >= 20:
        input_image.save(output_path, format="JPEG", quality=quality)
        if os.path.getsize(output_path) <= 2 * max_size_kb * 1024:
            break
        quality -= 5
    return output_path

def generate_thumbnail_with_multiple_keys(topic: str = "{topic}", max_retries=50, wait_seconds=5):

   
    # prompt = (
    #         f"Create a highly attractive and clickable thumbnail for a video titled '{topic}'. "
    #         "The thumbnail should have vibrant colors, bold and readable text displaying the title, "
    #         "an expressive human face showing excitement or surprise, "
    #         "and relevant high-quality imagery that visually represents the topic clearly. "
    #         "Include dynamic backgrounds, subtle 3D effects, and modern design elements to grab attention. "
    #         "Make sure the thumbnail looks professional, clean, and irresistible to click. "
    #         "Thumbnail must be mobile-fit in vertical 9:16 ratio (1080x1920). Text should be short and visually optimized."
    #     )
    # prompt = (
    #     f"Create a highly attractive and clickable thumbnail for a music video titled '{topic}'. "
    #     "The thumbnail should feature vibrant and moody colors that match the vibe of the song, "
    #     "bold and stylish text with the song title (short and impactful), "
    #     "a striking human figure or artist silhouette showing strong emotion or performance energy, "
    #     "and visually rich imagery that reflects the genre or theme of the music. "
    #     "Incorporate modern design elements like glow effects, dynamic lighting, and clean overlays to make it pop. "
    #     "Ensure the thumbnail is vertical (9:16 ratio, 1080x1920) and optimized for mobile screens. "
    #     "It must look professional, emotionally engaging, and impossible to ignore."
    # )
    prompt = (
        f"Create a bright, fun, and highly clickable thumbnail for a kids music video titled '{topic}'. "
        "The thumbnail should feature cheerful and colorful visuals that immediately appeal to babies and toddlers. "
        "Use large, bubbly, and playful text for the song title (short and easy to read). "
        "Include cute cartoon characters, smiling babies, or animated animals showing joy and energy. "
        "Use vibrant colors like yellow, blue, pink, and green, with a soft glow and friendly design elements. "
        "Add fun imagery like stars, balloons, toys, or musical notes that reflect a happy and playful mood. "
        "The thumbnail should be vertical (9:16 ratio, 1080x1920), optimized for mobile viewing, "
        "and professionally designed to feel safe, engaging, and impossible for parents and kids to ignore."
    )


    # prompt = (
    #     f"Create a highly attractive and clickable YouTube thumbnail for a video titled '{topic}'. "
    #     "The thumbnail should use only visual elements — no text at all. "
    #     "Use vibrant colors, expressive human faces showing strong emotions like excitement or surprise, "
    #     "and high-quality, relevant imagery that clearly represents the topic visually. "
    #     "Incorporate dynamic backgrounds, subtle 3D effects, and modern design elements to grab attention. "
    #     "Make sure the thumbnail looks professional, clean, and irresistible to click. "
    #     "The thumbnail must be mobile-optimized in a vertical 9:16 ratio (1080x1920). Do not include any YouTube icons or written text."
    # )

    # prompt = (
    #     f"Create a hyper-clickable, emotionally intense YouTube thumbnail for a video titled '{topic}'. "
    #     "Only use visual elements — absolutely no text. The thumbnail must grab attention instantly, within milliseconds. "
    #     "Focus on expressive human faces with exaggerated emotions like shock, fear, awe, or intense curiosity. "
    #     "Use bold, contrasting colors and dramatic lighting to draw the eye. Show a powerful visual moment that creates instant mystery or suspense — the viewer should immediately wonder: 'What’s going on here?' "
    #     "Include dynamic movement or tension (e.g., pointing, reaching, falling, glowing eyes, or something mid-action). "
    #     "Use cinematic composition, shallow depth of field, and a clean, high-contrast layout. "
    #     "It must feel urgent, extreme, and irresistible to click — especially on mobile. Output should be 1080x1920, vertical (9:16), and must not contain any logos, watermarks, or text."
    # )
    # prompt = (
    #     f"Design a highly clickable, emotionally explosive YouTube thumbnail for a video titled '{topic}'. "
    #     "Use **only visual elements** — shoukd use title fro text overlay effectively with bold capital letters with best color contrast but size must not make visual seem cluttered. so text size msut be accordingly . "
    #     "Focus on **expressive, close-up human faces** with exaggerated emotions like shock, awe, fear, or extreme curiosity. "
    #     "Incorporate dramatic lighting** to create tension. "
    #     "The image should suggest a **powerful, mysterious event or moment** — something that makes the viewer think, 'What the hell is going on here?' within milliseconds. "
    #     "Include dynamic motion or visual tension  "
    #     "Use **cinematic techniques**: shallow depth of field, center-weighted or rule-of-thirds composition, and clean foreground/background separation. "
    #     "Ensure the visual feels urgent, extreme, and impossible to ignore — especially when seen on mobile. "
    #     "Final output: 1080x1920 (9:16 vertical), photo-realistic"
    # )



    disabled_keys_today = load_disabled_keys()

    for attempt in range(max_retries):
        available_keys = [
            k for k in api_keys
            if k not in disabled_keys_today
            and not has_exceeded_daily_limit(k, limit=100)
            and not has_exceeded_minute_limit(k, limit=10)
        ]

        if not available_keys:
            print("❌ All API keys are either disabled or exceeded usage limits.")
            return None

        key = random.choice(available_keys)
        print(f"🔁 Trying API key ending with: {key[-6:]}")

        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
            )

            increment_usage(key)

            parts = response.candidates[0].content.parts if hasattr(response, "candidates") else []
            if not parts:
                print("⚠️ No parts in the Gemini response.")
                save_disabled_key(key)
                disabled_keys_today.add(key)
                time.sleep(wait_seconds)
                continue

            for part in parts:
                if hasattr(part, "text") and part.text:
                    print("📝 Gemini Response Text:", part.text)

                elif hasattr(part, "inline_data") and part.inline_data:
                    data = part.inline_data.data
                    image_data = base64.b64decode(data) if isinstance(data, str) else data

                    if image_data:
                        try:
                            image = Image.open(BytesIO(image_data)).convert("RGB")
                            image = resize_to_1080x1920_stretch(image)
                            os.makedirs("temp/thumbnails", exist_ok=True)
                            output_path = f"temp/thumbnails/thumbnail_{int(time.time())}.jpg"
                            compress_thumbnail(image, output_path)
                            print(f"✅ Thumbnail saved and compressed: {output_path}")
                            return output_path
                        except Exception as e:
                            print("❌ Error processing image:", e)
                            return None

            print("⚠️ No valid image found in this response.")
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


# Get trending topic from Google Trends using SerpAPI



UPLOAD_LIMIT = 11
JSON_FILE = "UPLOAD_STATUS.json"


import requests
from PIL import Image, ImageOps
from io import BytesIO
import os
from moviepy.editor import ImageClip, concatenate_videoclips

# Config
PEXELS_API_KEY = "DGhCtAB83klpCIv5yq5kMIb2zun7q67IvHJysvW4lInb0WVXaQF2xLMu" # Replace with your Pexels API key

NUM_IMAGES = 21
VIDEO_DURATION = 5
PER_IMAGE_DURATION = VIDEO_DURATION / NUM_IMAGES
RESOLUTION = (1920, 1080)
TMP_DIR = "pexels_images"
OUTPUT_VIDEO = "outputcuts.mp4"

def is_near_resolution(size, target=(1920, 1080), tol=0.1):
    w, h = size
    tw, th = target
    return abs(w - tw)/tw <= tol and abs(h - th)/th <= tol

def clear_temp_folder(folder):
    if os.path.exists(folder):
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    else:
        os.makedirs(folder)
import hashlib

def get_cache_folder_for_topic(topic):
    # Generate a simple hash folder name for topic (to avoid illegal characters in folder name)
    topic_hash = hashlib.md5(topic.encode('utf-8')).hexdigest()
    return os.path.join("pexels_cache", topic_hash)

def load_images_from_cache(cache_folder, count):
    if not os.path.exists(cache_folder):
        return []
    files = sorted([f for f in os.listdir(cache_folder) if f.endswith(".jpg")])
    if len(files) < count:
        return []
    return [os.path.join(cache_folder, f) for f in files[:count]]

def fetch_pexels_images(query, count):
    print("Fetching images from Pexels...")
    headers = {"Authorization": PEXELS_API_KEY}
    collected_urls = []
    page = 1

    while len(collected_urls) < count:
        url = f"https://api.pexels.com/v1/search?query={query}&per_page=15&page={page}"
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Pexels API Error: {response.status_code} - {response.text}")

        data = response.json()
        photos = data.get("photos", [])

        if not photos:
            break  # No more results

        for photo in photos:
            if len(collected_urls) >= count:
                break
            collected_urls.append(photo['src']['original'])

        page += 1

    if len(collected_urls) < count:
        print(f"Warning: Only {len(collected_urls)} images available from Pexels.")

    return collected_urls



def prepare_images(urls, folder, target_res):
    os.makedirs(folder, exist_ok=True)
    paths = []

    for i, url in enumerate(urls):
        try:
            print(f"Downloading image {i + 1}/{len(urls)}...")
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")

            # Resize and crop to exactly target_res without black borders
            img = ImageOps.fit(img, target_res, Image.LANCZOS, centering=(0.5, 0.5))


            path = os.path.join(folder, f"img_{i:02d}.jpg")
            img.save(path, quality=95)
            paths.append(path)
        except Exception as e:
            print(f"Error processing image {i + 1}: {e}")
    
    return paths



from Wav2Lip.inference import parser, run_inference

import sys
sys.path.append('Wav2Lip')



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



# Initialize JSON storage
def initialize_json():
    today = datetime.now().strftime("%Y-%m-%d")
    data = {"date": today, "count": UPLOAD_LIMIT}
    with open(JSON_FILE, "w") as f:
        json.dump(data, f)

# Load or reset upload status
def get_upload_status():
    if not os.path.exists(JSON_FILE):
        initialize_json()

    try:
        with open(JSON_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print("⚠️ JSON file corrupted or empty. Reinitializing...")
        initialize_json()
        with open(JSON_FILE, "r") as f:
            data = json.load(f)

    today = datetime.now().strftime("%Y-%m-%d")

    if data["date"] != today:
        print("🔄 New day. Resetting count.")
        data = {"date": today, "count": UPLOAD_LIMIT}
        with open(JSON_FILE, "w") as f:
            json.dump(data, f)

    return data


# Save updated count
def save_upload_status(count):
    today = datetime.now().strftime("%Y-%m-%d")
    with open(JSON_FILE, "w") as f:
        json.dump({"date": today, "count": count}, f)

def retry_infinite(delay=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    print(f"❌ function failed (Attempt {attempt}): {e}")
                    print(f"🔁 Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry_infinite(delay=5)
def get_trending_topic():
    params = {
        "engine": "google_trends_trending_now",
        "geo": "US",
        "api_key": SERP_API_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    trending = results.get("trending_searches", [])
    return trending[0]["query"] if trending else "World News"

@retry_infinite(delay=5)
def generate_youtube_script(topic):
    prompt = f"""
    write youtube script for {topic} in narrating style without intro and outro, be concise and precise straight to the point only content of topic nothing else, dont use bold text to emphasize or any mark like **, i just want plain normal format , it should have just script and nothing else mind it, script   video should be engaging and valuable to viewers, focus on quality and engagement,  i want video to be viral and generate infinite money with it so make script that good. script must be in paragraph format(required dont forget it mind it) dont write in bold anything script should be in paragraph format against youtube script word plainly and nothing else in it as i have to fed it to my ai program so making it understand would be easy so just assign script plainly against youtube script word,  script length msut be of max 1 minute only or max 11 sentences only and not more than this criteria, script msut be creative and engaging mind it i need it okay.  mind it it should bes seo optimised
    """
    response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
    full_text = response.text.strip()

    # Extract only the part after the phrase 'youtube script'
    if 'youtube script' in full_text.lower():
        # Find the first occurrence regardless of casing
        index = full_text.lower().index('youtube script')
        script = full_text[index + len('youtube script'):].strip()
        return script
    else:
        return full_text  # fallback if 'youtube script' not found

@retry_infinite(delay=5)    
def get_topic_from_script(script):
    prompt = f"""
    You are a YouTube content assistant. Given the following video script, generate a concise and relevant YouTube video title that clearly describes the main topic of the script.

    Script:
    \"\"\"
    {script}
    \"\"\"

    Only return the title as plain text without quotes or extra text.
    """
  
    response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
    return response.text.strip()

def mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to WAV using ffmpeg"""
    if os.path.exists(wav_path):
        os.remove(wav_path)
        print(f"🗑️ Deleted existing WAV file: {wav_path}")

    try:
        ffmpeg.input(mp3_path).output(wav_path, ar=16000, ac=1).run(overwrite_output=True)
        print(f"✅ MP3 to WAV Conversion complete: {wav_path}")
    except ffmpeg.Error as e:
        print("❌ FFmpeg error during mp3_to_wav:")
        print(e.stderr.decode())



def wav_to_mp3(wav_path, mp3_path):
    """Convert WAV to MP3 using ffmpeg"""
    if os.path.exists(mp3_path):
        os.remove(mp3_path)
        print(f"🗑️ Deleted existing MP3 file: {mp3_path}")

    try:
        ffmpeg.input(wav_path).output(mp3_path, acodec='libmp3lame').run(overwrite_output=True)
        print(f"✅ WAV to MP3 Conversion complete: {mp3_path}")
    except ffmpeg.Error as e:
        print("❌ FFmpeg error during wav_to_mp3:")
        print(e.stderr.decode())


# === Extract Keywords with SpaCy === #
def extract_keywords(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ("NOUN", "PROPN", "ADJ")]

# === Fallback Search Queries (3) === #

@retry_infinite(delay=5)
def fallback_search_query(sentence, user_topic):
    prompt = f"Give a short search query to find a stock video for: '{sentence}', just plainly text query not in bold, give 1  query search term only, and nothing else,  must relate to topic so that video clips are used which are relevent to topic pls kindly create search terms which are related to topic and current part dont get offshore, user topic is {user_topic}"
    response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
    return [q.strip() for q in response.text.strip().split(",")]

API_KEY = '50331047-e9be991568dc6ca136acd003b'  # Replace with your actual Pixabay API key

@retry_infinite(delay=5)
def search_pixabay_videos(query, per_page=200, max_results=15):  # lowered per_page for quicker tests
    url = 'https://pixabay.com/api/videos/'
    page = 1
    found_clips = []

    while len(found_clips) < max_results:
        params = {'key': API_KEY, 'q': query, 'per_page': per_page, 'page': page}
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"❌ Pixabay Error: {response.status_code} - {response.text}")
            break

        results = response.json().get('hits', [])
        if not results:
            print(f"❌ Pixabay: No more results on page {page} for '{query}'")
            break

        print(f"🔎 Pixabay Page {page}: Found {len(results)} videos for '{query}'")

        for i, clip in enumerate(results):
            videos = clip.get("videos", {})
            for quality_key, quality in videos.items():
                width = quality.get("width", 0)
                height = quality.get("height", 0)
                video_url = quality.get("url", "NO_URL")

                if abs(width - 1920) <= 200 and abs(height - 1080) <= 200:
                    clip["video_files"] = [quality]
                    clip["width"] = width
                    clip["height"] = height
                    clip["source"] = "pixabay"

                    if video_url not in [c["video_files"][0]["url"] for c in found_clips]:
                        found_clips.append(clip)

                    break  # Exit inner loop once 1 quality match is found

        page += 1

    return found_clips



import requests

import requests
import json
@retry_infinite(delay=5)
def search_pexels_video(query, per_page=80, target_width=1920, target_height=1080, tolerance=200, max_clips=15):
    page = 1
    headers = {"Authorization": PEXELS_API_KEY}
    suitable_clips = []

    while True:
        params = {"query": query, "per_page": per_page, "page": page}
        response = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params)

        if response.status_code != 200:
            print(f"❌ Pexels Error: {response.status_code} - {response.text}")
            return suitable_clips

        results = response.json().get("videos", [])
        if not results:
            print(f"❌ Pexels: No more results on page {page} for '{query}'")
            break

        print(f"🔎 Pexels Page {page}: Found {len(results)} videos for '{query}'")

        if page == 1:
            print("Sample raw results for inspection:")
            print(json.dumps(results[:2], indent=2))

        for i, clip in enumerate(results):
            for file in clip["video_files"]:
                width = file.get("width", 0)
                height = file.get("height", 0)
                video_url = file.get("link", "NO_URL")

                print(f"Candidate {i}: URL={video_url}, Resolution={width}x{height}")

                if abs(width - target_width) <= tolerance and abs(height - target_height) <= tolerance:
                    print(f"✅ Suitable video found at {video_url} with resolution {width}x{height}")

                    clip["video_files"] = [file]
                    clip["width"] = width
                    clip["height"] = height
                    clip["source"] = "pexels"

                    suitable_clips.append(clip)
                    break  # No need to check other file resolutions for this clip

            if len(suitable_clips) >= max_clips:
                print(f"🎯 Collected {max_clips} suitable clips. Stopping search.")
                return suitable_clips

        print(f"➡️ No enough clips yet, moving to page {page + 1}...")
        page += 1

    return suitable_clips




# === Search Pexels for a Query === #



import requests



import requests
# Replace with your actual Pixabay API key

import os

@retry_infinite(delay=5)
def download_videos1(video_url, download_path):
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    try:
        print(f"📥 Downloading video from: {video_url}")
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Downloaded: {download_path}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download video: {e}")




# === Find 1 Best Clip Per Search Term, avoiding duplicates === #

@retry_infinite(delay=5)
def find_one_video_clips(sentence, used_video_urls, user_topic, max_clips=15):
    print(f"🔍 Searching multiple clips for: {sentence}")
    queries = fallback_search_query(sentence, user_topic)  # Single query expected in your case

    def is_valid(clip_url, width, height):
        return (
            abs(width - 1920) <= 200 and
            abs(height - 1080) <= 200 and
            clip_url not in used_video_urls
        )

    def process_pexels(query):
        print(f"🔎 [Pexels] Query: {query}")
        results = []
        for clip in search_pexels_video(query):
            video_url = clip["video_files"][0]["link"]
            if is_valid(video_url, clip["width"], clip["height"]):
                clip["source"] = "pexels"
                results.append(clip)
                if len(results) >= max_clips:
                    break
        return results

    def process_pixabay(query):
        print(f"🔎 [Pixabay] Query: {query}")
        results = []
        for clip in search_pixabay_videos(query, per_page=200):
            try:
                video = clip["videos"]["medium"]
                video_url = video["url"]
                if is_valid(video_url, video["width"], video["height"]):
                    results.append({
                        "video_files": [{"link": video_url}],
                        "width": video["width"],
                        "height": video["height"],
                        "source": "pixabay"
                    })
                    if len(results) >= max_clips:
                        break
            except KeyError:
                continue
        return results

    collected = []

    for query in queries:
        collected += process_pexels(query)
        if len(collected) >= max_clips:
            return collected[:max_clips]

        collected += process_pixabay(query)
        if len(collected) >= max_clips:
            return collected[:max_clips]

        # Try keywords from query if not enough
        keywords = extract_keywords(query)
        for keyword in keywords:
            collected += process_pexels(keyword)
            if len(collected) >= max_clips:
                return collected[:max_clips]

            collected += process_pixabay(keyword)
            if len(collected) >= max_clips:
                return collected[:max_clips]

    print("❌ Not enough suitable clips found.")
    return collected[:max_clips]







    
 # List of up to 3 unique clips, one per query



# === Download Video === #
@retry_infinite(delay=5)
def download_video(url, filename):
    response = requests.get(url, stream=True)
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return filename

from gtts import gTTS
from moviepy.editor import AudioFileClip
import subprocess

# === Generate Audio === #
@retry_infinite(delay=5)
def generate_audio(text, filename):
    # Generate raw gTTS audio
    tts = gTTS(text=text, lang='en')
    tts.save(filename)

    # Speed up using ffmpeg atempo filter
    sped_up_filename = filename.replace('.mp3', '_1.4x.mp3')
    subprocess.run([
        "ffmpeg", "-y", "-i", filename,
        "-filter:a", "atempo=1.4", "-vn", sped_up_filename
    ], check=True)

    return AudioFileClip(sped_up_filename)




from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip, vfx
@retry_infinite(delay=5)
def create_scene(text, idx, used_video_urls, user_topic):
    print(f"\n🎬 Creating Scene {idx + 1}")
    
    video_clips_data = find_one_video_clips(text, used_video_urls, user_topic, max_clips=15)

    if not video_clips_data:
        print("❌ No clips found.")
        return None, []

    try:
        audio_path = f"audio/scene_{idx}.mp3"
        
        # Generate and speed-up audio
        audio_clip = generate_audio(text, audio_path)

        # Get duration of sped-up audio
        audio_duration = audio_clip.duration

    except Exception as e:
        print(f"❌ Audio generation failed: {e}")
        return None, []


    clips = []
    global new_used_urls
    new_used_urls = []
    
    for i, clip_data in enumerate(video_clips_data):
        try:
            video_url = clip_data["video_files"][0]["link"]
            if video_url in used_video_urls:
                continue

            source = clip_data.get("source", "unknown")
            video_path = f"video_creation/scene_{idx}_{i}.mp4"

            if source == "pixabay":
                download_videos1(video_url, video_path)
            else:
                download_video(video_url, video_path)

            clip = VideoFileClip(video_path).without_audio()

            # Resize and crop
            target_aspect = 1920 / 1080
            actual_aspect = clip.w / clip.h
            if abs(actual_aspect - target_aspect) < 0.01:
                clip = clip.resize((1920, 1080))
            elif actual_aspect > target_aspect:
                clip = clip.resize(height=1080)
                x_center = clip.w / 2
                clip = clip.crop(x1=x_center - 960, x2=x_center + 960, y1=0, y2=1080)
            else:
                clip = clip.resize(width=1920)
                y_center = clip.h / 2
                clip = clip.crop(x1=0, x2=1920, y1=y_center - 540, y2=y_center + 540)

            clips.append(clip.set_fps(30))
            new_used_urls.append(video_url)

            if sum(c.duration for c in clips) >= audio_duration:
                break

        except Exception as e:
            print(f"⚠️ Error processing {clip_data.get('video_files')[0]['link']}: {e}")
            continue

    if not clips:
        print("❌ All clip processing failed.")
        return None, []

    # Concatenate and match audio length
    final_video = concatenate_videoclips(clips, method="compose")
    if final_video.duration > audio_duration:
        final_video = final_video.subclip(0, audio_duration)
    else:
        speed_factor = final_video.duration / audio_duration
        final_video = final_video.fx(vfx.speedx, factor=speed_factor).set_duration(audio_duration)

    used_video_urls.update(new_used_urls)

    from moviepy.editor import TextClip, CompositeVideoClip
    subtitle = (
        TextClip(text, fontsize=60, color="white", font="Arial-Bold", size=final_video.size, method="caption")
        .set_position(("center", "bottom"))
        .set_duration(audio_duration)
        .fadein(1)
    )

    final_clip = CompositeVideoClip([final_video, subtitle])
    final_clip = final_clip.set_duration(audio_duration).set_audio(audio_clip)

    return final_clip, new_used_urls







import subprocess



import os
import subprocess
from datetime import datetime
from nltk.tokenize import sent_tokenize
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeAudioClip, AudioFileClip


from moviepy.editor import AudioFileClip, CompositeAudioClip, ImageClip, concatenate_videoclips

def create_fast_cut_clip_from_images(image_paths, total_duration=5, resolution=(1920,1080)):
    per_image_duration = total_duration / len(image_paths)
    tick_sound = AudioFileClip("/Users/uday/Downloads/VIDEOYT/analog-camera-shutter-96604_z7Dhy2kD.mp3").volumex(0.1)
    clips = []

    for i, path in enumerate(image_paths):
        clip = ImageClip(path).set_duration(per_image_duration).fadein(0.1).fadeout(0.05).resize(resolution)
        tick_audio = tick_sound.subclip(0, min(0.17, tick_sound.duration))
        clip = clip.set_audio(tick_audio)
        clips.append(clip)

    fast_cut_clip = concatenate_videoclips(clips, method="compose")

 

    return fast_cut_clip


def create_video_from_script(script, user_topic, include_disclaimer=True):
    from nltk.tokenize import sent_tokenize
    from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip

    sentences = sent_tokenize(script)
    used_video_urls = set()
    scene_clips = []

    output_dir = "video_created"
    os.makedirs(output_dir, exist_ok=True)

    # Add disclaimer video if available
    disclaimer_path = "/Users/uday/Downloads/VIDEOYT/disclaimer_video.mp4"
    if include_disclaimer and os.path.exists(disclaimer_path):
        disclaimer_clip = VideoFileClip(disclaimer_path).without_audio()
        scene_clips.append(disclaimer_clip)
    elif include_disclaimer:
        print(f"⚠️ Disclaimer video not found at: {disclaimer_path}. Skipping disclaimer.")

    # After disclaimer, add the 5-second fast cut video from Pexels images
    cache_folder = get_cache_folder_for_topic(user_topic)
    images = load_images_from_cache(cache_folder, NUM_IMAGES)

    if not images:
        print("No sufficient cached images, fetching from Pexels...")
        clear_temp_folder(cache_folder)
        urls = fetch_pexels_images(user_topic, 21)
        images = prepare_images(urls, cache_folder, RESOLUTION)

    fast_cut_clip = create_fast_cut_clip_from_images(images, total_duration=5, resolution=RESOLUTION)
    scene_clips.append(fast_cut_clip)  # mute if you want silent here, or keep audio by removing .without_audio()

    # Process scripted sentences into clips
    all_used_urls = set()

    for idx, sentence in enumerate(sentences):
        scene_clip, scene_urls = create_scene(sentence, idx, used_video_urls, user_topic)
        if scene_clip is not None:
            scene_clips.append(scene_clip)
        all_used_urls.update(scene_urls)

    print("All used URLs:", all_used_urls)

    if not scene_clips:
        print("❌ No scenes could be created. Cannot generate final video.")
        return None, None

    # Check for valid clips
    for i, clip in enumerate(scene_clips):
        if not hasattr(clip, 'duration'):
            print(f"❌ Invalid clip at index {i}: {clip}")
            return None, None

    # Concatenate all video clips
    final_clip = concatenate_videoclips(scene_clips, method="compose")
    output_path = os.path.join(output_dir, "final_video_01.mp4")

    # Background music
    from moviepy.audio.fx.all import audio_loop
    # Background music
    bg_music_raw = AudioFileClip("/Users/uday/Downloads/VIDEOYT/Cybernetic Dreams.mp3").volumex(0.03)
    bg_music_looped = audio_loop(bg_music_raw, duration=final_clip.duration).set_start(5)

    # Mix with existing audio (tick sounds, etc.)
    if final_clip.audio:
        final_audio = CompositeAudioClip([final_clip.audio.set_duration(final_clip.duration), bg_music_looped])
    else:
        final_audio = bg_music_looped


    final_clip = final_clip.set_audio(final_audio)

    # Write final video
    final_clip.write_videofile(
        output_path,
        codec="libx264",            # Video codec
        audio_codec="libmp3lame",   # Ensures .mp3-compatible audio encoding
        fps=30,
        audio=True,
        preset="ultrafast",
        threads=8
    )

    print(f"✅ Final video created at: {output_path}")

    # Return output_path and all_used_urls for use elsewhere
    return output_path







# @retry_infinite(delay=5)
# def generate_youtube_title(topic):
#     prompt = f"""
#     Create a highly clickable, viral YouTube video title for the topic: "{topic}".
#     Do not include any extra explanation or formatting. Just return the title only. it should be seo optimised mind it
#     """
#     response = gemini_model.generate_content(prompt)
#     return response.text.strip()

@retry_infinite(delay=5)
def generate_music_video_description(song_name):
    # prompt = f"""
# You are a YouTube SEO expert with access to real-time vidIQ Pro and TubeBuddy Boost data.

# Write a powerful, SEO-optimized YouTube video description for a music video titled: "{song_name}". The description must:

# - Start with an attention-grabbing hook that creates emotional impact or curiosity
# - Seamlessly include high-ranking music-related SEO keywords tied to the song's mood, genre, or artist type
# - Clearly convey the vibe, theme, or story of the music video to engage the viewer
# - Naturally motivate viewers to like, comment, and subscribe without sounding robotic
# - Use a single, engaging paragraph format (no bullet points, no line breaks, no sections)
# - End with exactly 15 high-performing YouTube music-related hashtags

# 🚨 Hashtag Rules:
# - Must be **1 word** only
# - Must be **ultra-high volume**, **high CTR**, and **SEO-optimized**
# - Must relate to music, trending songs, viral audio, beats, rap, chill, emotional, or dance
# - Output hashtags **in-line at the end of the paragraph**, comma-separated
# - Do **not** include formatting hints, lists, or explanation

# ⚠️ Output format: Return ONLY the final viewer-ready description and hashtags — as one clean paragraph.
#     """
    prompt = f"""
    You are a YouTube SEO expert with access to real-time vidIQ Pro and TubeBuddy Boost data.

    Write a powerful, SEO-optimized YouTube video description for a music video titled: "{song_name}". The description must:

    - Start with an attention-grabbing hook that creates emotional impact or curiosity
    - Seamlessly include high-ranking music-related SEO keywords tied to the song's mood, genre, or artist type
    - Clearly convey the vibe, theme, or story of the music video to engage the viewer
    - Naturally motivate viewers to like, comment, and subscribe without sounding robotic
    - Use a single, engaging paragraph format (no bullet points, no line breaks, no sections)
    - End with exactly 15 high-performing YouTube music-related hashtags

    🚨 Hashtag Rules:
    - Must be **1 word** only
    - Must be **ultra-high volume**, **high CTR**, and **SEO-optimized**
    - Must relate to music, trending songs, viral audio, beats, rap, chill, emotional, or dance
    - Output hashtags **in-line at the end of the paragraph**, comma-separated
    - Do **not** include formatting hints, lists, or explanation

    ⚠️ Output format: Return ONLY the final viewer-ready description and hashtags — as one clean paragraph.
    """

    response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
    return response.text.strip()



@retry_infinite(delay=5)
def generate_music_video_tags(song_name):
#     prompt = f"""
# You are a YouTube SEO expert with access to real-time vidIQ Pro and TubeBuddy Boost data.

# For the music video titled "{song_name}", generate exactly 20 ultra-high-performing YouTube tags.

# Rules:
# - Each tag must be **1 word only**
# - Must be high-volume, high-CTR, SEO-optimized
# - Must relate to trending music, viral songs, rap, chill, emotional, beats, dance, pop, or the overall vibe of music videos
# - Tags must be highly relevant to helping this song rank and get discovered
# - Output only as a **comma-separated list**, no quotes, no line breaks, no formatting, and under 500 characters
#     """
    prompt = f"""
    You are a YouTube SEO expert with access to real-time vidIQ Pro and TubeBuddy Boost data.

    For the music video titled "{song_name}", generate exactly 20 ultra-high-performing YouTube tags.

    Rules:
    - Each tag must be **1 word only**
    - Tags must be trending, high-search-volume, high-CTR, and SEO-optimized
    - Must relate to music, viral songs, rap, chill, emotional, beats, dance, pop, or the overall vibe of popular music videos
    - Tags must help this song get discovered, rank better, and go viral
    - Output only a **comma-separated list**, no quotes, no line breaks, no extra text, and keep the output under 500 characters
    """


    response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
    tag_string = response.text.strip()
    tag_list = [tag.strip() for tag in tag_string.split(',')]

    # Limit to 500 characters total
    limited_tags = []
    total_length = 0
    for tag in tag_list:
        tag_len = len(tag)
        if total_length + tag_len + (1 if limited_tags else 0) <= 500:
            limited_tags.append(tag)
            total_length += tag_len + (1 if limited_tags else 0)
        else:
            break

    return limited_tags


import ast

# @retry_infinite(delay=5)
# def clean_tags_with_gemini(raw_tags):
#     prompt = f"""
#     You are an assistant that cleans and formats YouTube video tags.

#     Here are the original tags:
#     {', '.join(raw_tags)}

#     Rules:
#     - Remove duplicates.
#     - Avoid overly long or repetitive phrases.
#     - Limit the total character count to under 500 characters.
#     - Respond ONLY with the cleaned tags list, comma-separated. Do not add any extra text any other than that in one line.
#     """

#     response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
#     try:
#         tag_string = response.text.strip()
#         cleaned_tags = [tag.strip() for tag in tag_string.split(',') if tag.strip()]
#         return cleaned_tags
#     except Exception as e:
#         print("❌ Error parsing Gemini response:", e)
#         print("Raw response:", response.text)
#         return []


@retry_infinite(delay=5)
def get_category_id_from_gemini(topic):
    prompt = f"""
    Given the YouTube video topic: "{topic}", return the most appropriate YouTube Category Name and ID from this list:

    1 - Film & Animation  
    2 - Autos & Vehicles  
    10 - Music  
    15 - Pets & Animals  
    17 - Sports  
    19 - Travel & Events  
    20 - Gaming  
    22 - People & Blogs  
    23 - Comedy  
    24 - Entertainment  
    25 - News & Politics  
    26 - Howto & Style  
    27 - Education  
    28 - Science & Technology  
    29 - Nonprofits & Activism  

    Only return the output in this format:
    Category Name: <name>  
    Category ID: <id>
    """


    response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
    return response.text


def extract_category_id(text):
    match = re.search(r'Category ID:\s*(\d+)', text)
    return int(match.group(1)) if match else 27  # Default to Education if not found






# --- Functions ---
# --- New Function to Generate Relevant Search Term ---
@retry_infinite(delay=5)
def generate_search_term(topic):
    prompt = f"Given the YouTube video topic '{topic}', suggest a short, relevant visual keyword or phrase for finding an image background. Limit to 3-5 words, no punctuation, just a plain image search phrase only one line and nothing else : keywords or phrase  just onloy that and nothing else mind it."
    response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
    return response.text.strip()

@retry_infinite(delay=5)
def generate_title_from_topic1(topic):
    prompt = f"Create a catchy YouTube video thumbnail title for this topic: '{topic}'  one line title which seo optimised and nothing else okay i ahve to fed to my progrma so it should be clena and precise dont use symbols or icons or emojis, cerate catchy one, use punctuation marks properly and highly to emphasize, and it should have max upto 5 or 6 words not more that that title shoul dnot include word shorts or #shorts like that pls mind it,  mind it it should bes seo optimised"
    response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
    return response.text.strip()

def resize_and_crop_to_1920x1080(img):
    target_ratio = 1080 / 1920
    img_ratio = img.width / img.height

    if img_ratio > target_ratio:
        # Image is too wide
        new_height = 1920
        new_width = int(1920 * img_ratio)
    else:
        # Image is too tall or exact fit
        new_width = 1080
        new_height = int(1080 / img_ratio)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Crop center
    left = (img.width - 1080) // 2
    top = (img.height - 1920) // 2
    right = left + 1080
    bottom = top + 1920

    return img.crop((left, top, right, bottom))

@retry_infinite(delay=5)
def search_pexels_image(query):
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": 10}
    resp = requests.get(url, headers=headers, params=params)
    photos = resp.json().get("photos", [])

    def is_aspect_ratio_close(photo):
        width, height = photo["width"], photo["height"]
        ratio = width / height
        return 1.7 < ratio < 1.9  # Close to 16:9

    # Try to find a 16:9 image
    for photo in photos:
        if is_aspect_ratio_close(photo):
            return photo["src"]["original"]

    # Fallback to first image if none are 16:9
    return photos[0]["src"]["original"] if photos else None

@retry_infinite(delay=5)
def search_google_image(query):
    params = {
        "engine": "google",
        "q": query,
        "tbm": "isch",
        "api_key": SERP_API_KEY,
        "num": 10  # get multiple images
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    images = results.get("images_results", [])

    def is_aspect_ratio_close(img_obj):
        try:
            url = img_obj["original"]
            img = download_image1(url)
            ratio = img.width / img.height
            # Check if ratio is approx 16:9 and size is decent
            if img.width >= 1080 and img.height >= 1920 and 1.7 < ratio < 1.9:
                return True, url
            else:
                return False, None
        except Exception:
            return False, None

    for img_obj in images:
        ok, url = is_aspect_ratio_close(img_obj)
        if ok:
            return url  # Return first good image URL

    # Fallback
    if images:
        return images[0]["original"]
    return None

@retry_infinite(delay=5)
def download_image1(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def get_dominant_color(img, resize_scale=150):
    """Resize image and get the most common RGB color."""
    small_img = img.resize((resize_scale, resize_scale))
    pixels = np.array(small_img).reshape(-1, 3)
    most_common = Counter(map(tuple, pixels)).most_common(1)[0][0]
    return most_common  # returns (R, G, B)

def invert_color(rgb):
    """Invert an RGB color."""
    return tuple(255 - v for v in rgb)

from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

def get_readable_color(rgb_color):
    # Determine if black or white will be more readable on the given background color
    r, g, b = rgb_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 186 else (255, 255, 255)
from PIL import Image, ImageDraw, ImageFont
import os




from PIL import Image, ImageDraw, ImageFont
import os


# def create_thumbnail(topic):
  

#     print("🎯 Generating title...")
#     title = generate_title_from_topic1(topic).replace("shorts", "").replace("Shorts", "").strip()
#     for word in ["shorts", "Shorts", "#shorts", "SHORTS","#Shorts"]:
#         title = title.replace(word, "")
#     title = title.strip()

#     print("📝 Title:", title)

#     input_img_path = "/Users/uday/Downloads/Untitled design.png"
#     output_path = "/Users/uday/Downloads/VIDEOYT/final_thumbnail.png"

#     print("📥 Loading original image...")
#     img = Image.open(input_img_path).convert("RGBA")  # Preserve original size

#     print("🎨 Analyzing colors...")
#     dominant_color = get_dominant_color(img)
#     text_color = get_readable_color(dominant_color)
#     border_color = get_readable_color(text_color)

#     draw = ImageDraw.Draw(img)
#     font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
#     if not os.path.exists(font_path):
#         raise FileNotFoundError("⚠️ Font path not found")

#     # Dynamically fit font size
#     font_size = 100
#     max_width = img.width - 100
#     while font_size > 10:
#         font = ImageFont.truetype(font_path, font_size)
#         text_width = draw.textlength(title, font=font)
#         if text_width <= max_width:
#             break
#         font_size -= 2
#     else:
#         print("⚠️ Could not find suitable font size, skipping text.")
#         img.convert("RGB").save(output_path)
#         return

#     # Calculate position
#     text_bbox = draw.textbbox((0, 0), title, font=font)
#     text_w = text_bbox[2] - text_bbox[0]
#     text_h = text_bbox[3] - text_bbox[1]
#     x = (img.width - text_w) // 2
#     y = img.height - text_h - 60

#     # Add transparent overlay
#     overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
#     overlay_draw = ImageDraw.Draw(overlay)
#     padding = 40
#     overlay_draw.rectangle(
#         [x - padding, y - padding, x + text_w + padding, y + text_h + padding],
#         fill=(0, 0, 0, 160)
#     )
#     img = Image.alpha_composite(img, overlay)

#     # Add border text (shadow)
#     draw = ImageDraw.Draw(img)
#     for dx in range(-2, 3):
#         for dy in range(-2, 3):
#             if dx != 0 or dy != 0:
#                 draw.text((x + dx, y + dy), title, font=font, fill=border_color)

#     # Draw final text
#     draw.text((x, y), title, font=font, fill=text_color)

#     img.convert("RGB").save(output_path)
#     print(f"✅ Final thumbnail saved at: {output_path}")

from PIL import Image, ImageDraw, ImageFont
from collections import Counter
from io import BytesIO
import numpy as np
import requests
import os

def create_thumbnail(topic):
    print("🎯 Generating image search term...")
    search_term = generate_search_term(topic)
    print("🔍 Search term:", search_term)

    print("🔎 Searching for background image...")
    img_url = search_pexels_image(search_term)
    if not img_url:
        print("⚠️ Pexels failed, using Google Images...")
        img_url = search_google_image(search_term)
    if not img_url:
        raise RuntimeError("❌ No suitable image found.")

    print("📥 Downloading image...")
    img = download_image1(img_url)

    print("✂️ Resizing and cropping to 1920x1080...")
    img = resize_and_crop_to_1920x1080(img).convert("RGBA")

    print("🎯 Generating title...")
    title = generate_title_from_topic1(topic).strip()
    for word in ["shorts", "Shorts", "#shorts", "SHORTS", "#Shorts"]:
        title = title.replace(word, "")
    title = title.strip()
    print("📝 Title:", title)

    print("🎨 Analyzing colors...")
    dominant_color = get_dominant_color(img)
    text_color = get_readable_color(dominant_color)
    border_color = get_readable_color(text_color)

    draw = ImageDraw.Draw(img)
    font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError("⚠️ Font path not found")

    font_size = 100
    max_width = img.width - 100
    while font_size > 10:
        font = ImageFont.truetype(font_path, font_size)
        text_width = draw.textlength(title, font=font)
        if text_width <= max_width:
            break
        font_size -= 2

    # Positioning
    text_bbox = draw.textbbox((0, 0), title, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    x = (img.width - text_w) // 2
    y = img.height - text_h - 60

    print("🧱 Adding overlay...")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    padding = 40
    overlay_draw.rectangle(
        [x - padding, y - padding, x + text_w + padding, y + text_h + padding],
        fill=(0, 0, 0, 160)
    )
    img = Image.alpha_composite(img, overlay)

    print("🖋️ Drawing text with border...")
    draw = ImageDraw.Draw(img)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), title, font=font, fill=border_color)

    draw.text((x, y), title, font=font, fill=text_color)

    output_path = "/Users/uday/Downloads/VIDEOYT/final_thumbnail.png"
    img.convert("RGB").save(output_path)
    print(f"✅ Final thumbnail saved at: {output_path}")
    return output_path

# def trim_tags(tags, max_length=490):
#     final_tags = []
#     total_len = 0
#     for tag in tags:
#         if total_len + len(tag) + 2 <= max_length:
#             final_tags.append(tag)
#             total_len += len(tag) + 2  # +2 for comma and space
#         else:
#             break
#     return final_tags

import time
import googleapiclient.errors

# MAX_RETRIES = 10
# @retry_infinite(delay=5)
# def resumable_upload(request):
#     response = None
#     error = None
#     retry = 0

#     while response is None:
#         try:
#             print("Uploading...")
#             status, response = request.next_chunk()
#             if response:
#                 print(f"✅ Upload complete. Video ID: {response['id']}")
#                 return response
#         except (googleapiclient.errors.HttpError, IOError) as e:
#             error = e
#             retry += 1
#             if retry > MAX_RETRIES:
#                 print("❌ Upload failed after multiple retries.")
#                 return None
#             sleep_seconds = 2 ** retry
#             print(f"Unexpected error: {str(e)}")
#             print(f"🔁 Retrying in {sleep_seconds} seconds...")
#             time.sleep(sleep_seconds)

import logging

import re

def sanitize_text(text, max_length=5000):
    """
    Clean and truncate text for YouTube metadata.
    - Replace multiple whitespace/newlines with single space.
    - Remove problematic control characters.
    - Truncate to max_length.
    """
    # Remove control characters except line breaks (optional)
    text = re.sub(r'[\x00-\x09\x0b-\x0c\x0e-\x1f\x7f]', '', text)
    
    # Normalize whitespace & newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate to max_length
    if len(text) > max_length:
        text = text[:max_length]
    
    return text

# def sanitize_tags(tags, max_tag_length=30):
#     """
#     Clean tags to be valid YouTube tags.
#     - Remove empty strings.
#     - Truncate tags that are too long.
#     """
#     clean_tags = []
#     for tag in tags:
#         tag = tag.strip()
#         if not tag:
#             continue
#         if len(tag) > max_tag_length:
#             tag = tag[:max_tag_length]
#         clean_tags.append(tag)
#     return clean_tags


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("upload.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

import random
import google.auth.transport.requests
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

import json
import time
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
import random
import time
import socket
import logging
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
import requests

logger = logging.getLogger(__name__)

import random
import time
import socket
import requests
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload


import os
from googleapiclient.http import MediaFileUpload

# def get_chunk_size(file_path):
#     file_size = os.path.getsize(file_path)  # size in bytes
#     mb = 1024 * 1024
    
#     if file_size < 50 * mb:
#         return 50*mb
#     elif file_size < 200 * mb:
#         return 100*mb
#     else:
#         return 400*mb
from googleapiclient.http import MediaFileUpload


import os
import pickle
from googleapiclient.discovery import build

TOKEN_PATHS = [

    "/Users/uday/Downloads/VIDEOYT/token5.pickle"
]

def build_youtube_client(token_path):
    with open(token_path, "rb") as token_file:
        credentials = pickle.load(token_file)
    return build("youtube", "v3", credentials=credentials)

def is_token_usable(youtube):
    try:
        channel_response = youtube.channels().list(part="contentDetails", mine=True).execute()
        return channel_response is not None
    except Exception as e:
        print(f"❌ Token not usable or quota exceeded: {e}")
        return False

def get_available_token():
    for path in TOKEN_PATHS:
        try:
            youtube = build_youtube_client(path)
            if is_token_usable(youtube):
                print(f"✅ Using token: {path}")
                return youtube, path
        except Exception as e:
            print(f"⚠️ Failed to load token: {path} — {e}")
    return None, None

def load_video_topic_mappings(filepath):
    """
    Loads video path and topic mappings from a text file.
    Format per line: /path/to/video.mp4 | Topic
    Returns: List of tuples (video_path, topic)
    """
    mappings = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' not in line:
                continue
            video_path, topic = map(str.strip, line.split('|', 1))
            mappings.append((video_path, topic))
    return mappings

def remove_mapping_entry(video_path, mapping_file="file_topic_map.txt"):
    with open(mapping_file, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if not line.strip().startswith(video_path):  # Skip the line matching the video
            new_lines.append(line)

    with open(mapping_file, "w") as f:
        f.writelines(new_lines)

    print(f"✅ Mapping entry for '{video_path}' removed from {mapping_file}")



def get_urls_for_video(video_name, log_file="all_video_used_urls.txt"):
    urls = []
    with open(log_file, "r") as f:
        lines = f.readlines()

    collecting = False
    for line in lines:
        line = line.strip()
        if not line:
            collecting = False  # Stop when reaching a blank line
            continue
        if line == video_name:
            urls = []           # Clear in case video name appears multiple times
            collecting = True
            continue
        if collecting:
            urls.append(line)

    return urls





def remove_video_entry(video_name, log_file="all_video_used_urls.txt"):
    with open(log_file, "r") as f:
        lines = f.readlines()

    new_lines = []
    skip = False

    for line in lines:
        if line.strip() == video_name:
            skip = True  # Start skipping this entry
            continue
        if skip:
            if line.strip() == "":  # End of this entry
                skip = False
            continue
        new_lines.append(line)

    with open(log_file, "w") as f:
        f.writelines(new_lines)

    print(f"✅ Entry for '{video_name}' removed from {log_file}")

# import google.generativeai as genai

# # Set your Gemini API key
# genai.configure(api_key="AIzaSyA2Hj5phmEsqXBWqIGbZxQXxAzv129Zw1E")

def generate_title_with_hashtags(topic: str) -> str:
    # prompt = (
    #     f"Generate 3–5 trending, viral, SEO-optimized hashtags relevant to the topic: \"{topic}\" "
    #     f"Output only hashtags in one line, no explanation.  mind it it should bes seo optimised"
    # )
    prompt = (
        f"Generate 3–5 trending, viral, and SEO-optimized hashtags for the topic: \"{topic}\". "
        f"Only output hashtags in a single line, comma-separated, no explanation or extra text. "
        f"Hashtags must be highly relevant, currently popular, and boost visibility on YouTube and social platforms."
    )

    
    response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")

    hashtags = response.text.strip().replace("\n", " ")
    final = f"{topic} {hashtags}"

    # Truncate safely to max 90 chars without cutting words
    if len(final) > 80:
        truncated = final[:80]
        # Cut back to last space to avoid breaking a word/hashtag
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated
    return final




def generate_description_with_scene_links(base_description, feedback_link, video_path):
    description = sanitize_text(base_description)
    description += f"\n\n📢 We'd love your feedback! Share your thoughts here 👉 {feedback_link}\n\n"

    # Try to load scene video URLs
    # urls = get_urls_for_video(video_path, log_file="all_video_used_urls.txt")

    # Only add scene video sources if URLs are found
    # if urls:
    #     description += "🎥 Scene Video Sources:\n"
    #     for i, url in urls:
    #         description += f"Scene {i} video: {url}\n"

    # Add music credit
    # description += (
    #     "\nCinematic Technology | Cybernetic Dreams by Alex-Productions\n"
    #     "https://youtu.be/NDYRjTti5Bw\n"
    #     "Music promoted by https://onsound.eu/\n"
    # )

    return description

def get_best_song_title_for_yt(original_title):
#     prompt = f"""
# You are a viral music title expert and YouTube SEO strategist. Your job is to generate one extremely clickable, emotional, and SEO-optimized YouTube title for a new song called: "{original_title}".

# 🎯 GOAL: The title should grab attention instantly, go viral, rank on YouTube, and emotionally connect with the audience. It must drive massive clicks and views.

# 📌 RULES:
# - Start with or include the song name (or slight remix)
# - Must sound like a **viral hit** or **must-listen track**
# - Add trending emotional/SEO keywords: "viral", "official", "2025", "heartbreaking", "emotional", "anthem", "sad", "hit", "must hear"
# - Keep it short: UNDER 12 words
# - Avoid quotes or unnecessary words
# - Make it sound powerful, epic, unforgettable

# 🔁 STYLE EXAMPLES:
# - "{original_title} – The Viral Anthem Everyone’s Obsessed With"
# - "{original_title} – Official Music Video That Broke the Internet"
# - "{original_title} – 2025’s Most Emotional Hit"
# - dont use "|" mind it, instaed use "-"
# Respond ONLY with the final title. No explanation.
#     """
    prompt = f"""
    You are a viral kids content expert and YouTube SEO strategist. Your job is to generate one extremely clickable, fun, and SEO-optimized YouTube title for a new kids song called: "{original_title}".

    🎯 GOAL: The title should grab attention instantly, go viral among parents and kids, rank on YouTube, and sound super fun and friendly. It must drive massive clicks and views.

    📌 RULES:
    - Start with or include the song name (or a fun remix of it)
    - Must sound like a **must-watch** or **viral favorite** for kids
    - Add trending kids-related SEO keywords: "fun", "cute", "baby song", "nursery rhyme", "2025", "new", "must watch", "learning", "kids hit", "cartoon", "toddler"
    - Keep it short: UNDER 12 words
    - Avoid quotes or unnecessary filler
    - Make it sound cheerful, playful, and memorable
    - DO NOT use "|" — use "-" instead

    🔁 STYLE EXAMPLES:
    - "{original_title} – Fun Baby Song Every Kid Loves"
    - "{original_title} – New Kids Rhyme 2025 Hit"
    - "{original_title} – Must Watch Cartoon Song for Toddlers"
    - "{original_title} – Viral Nursery Rhyme Kids Can’t Stop Singing"
    Respond ONLY with the final title. No explanation.
    """

    response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
    return response.text.strip()

@retry_infinite(delay=5)
def upload_video(file_path, topic):
    
    # thumbnail_path = generate_thumbnail_with_multiple_keys(topic)
    # topic=get_best_song_title_for_yt(topic)
    thumbnail_path = generate_thumbnail_with_multiple_keys(topic)
    
    youtube, token_path = get_available_token()
    if not youtube:
        print("❌ All tokens failed or have reached the limit.")
        return False

    feedback_link = "https://forms.gle/NLQ3gmdrsNU7DKev6"
    description = generate_music_video_description(topic)
    description = generate_description_with_scene_links(description, feedback_link, file_path)

    if "#shorts" not in description.lower():
        description += "\n\n#shorts"

    final_tags = generate_music_video_tags(topic)

  

    shorts_tags = {"shorts", "youtube shorts", "viral shorts"}
    current_tags_lower = {tag.lower() for tag in final_tags}
    for st in shorts_tags:
        if st not in current_tags_lower:
            final_tags.append(st)

    category_info = get_category_id_from_gemini(topic)
    category_id = extract_category_id(category_info)

    logger.info("📦 Preparing video upload...")
    logger.info(f"🎬 Title: {topic}")
    logger.info(f"📝 Description: {description}")
    logger.info(f"🏷️ Tags: {final_tags}")
    logger.info(f"📁 File: {file_path}")

    request_body = {
        "snippet": {
            "title": topic,
            "description": description,
            "categoryId": "10",
        },
        "status": {
            "privacyStatus": "public"
        }
    }

    if final_tags:
        request_body["snippet"]["tags"] = final_tags
    else:
        logger.warning("⚠️ No tags provided, skipping tags field.")

    import json
    print(json.dumps(request_body, indent=2))

    try:
        media = MediaFileUpload(file_path, chunksize=-1, resumable=True, mimetype="video/mp4")
    except Exception as e:
        logger.error(f"❌ Error preparing video for upload: {e}")
        return

    try:
        request = youtube.videos().insert(
            part="snippet,status",
            body=request_body,
            media_body=media
        )
    except Exception as e:
        logger.error(f"❌ Failed to initiate upload: {e}")
        return

    response = None
    retry = 0
    max_retries = 111111111

    while response is None:
        try:
            logger.info("🚀 Uploading...")
            status, response = request.next_chunk()

            if status:
                logger.info(f"📶 Upload progress: {int(status.progress() * 100)}%")

            if response and 'id' in response:
                video_id = response["id"]
                video_url = f"https://youtu.be/{video_id}"
                logger.info(f"✅ Video uploaded successfully: {video_url}")

                if thumbnail_path:
                    try:
                        logger.info("🖼️ Uploading thumbnail...")
                        youtube.thumbnails().set(
                            videoId=video_id,
                            media_body=MediaFileUpload(thumbnail_path, mimetype='image/jpeg')
                        ).execute()
                        logger.info("✅ Thumbnail uploaded successfully!")
                    except Exception as e:
                        logger.warning(f"❌ Thumbnail upload failed: {e}")

                # ✅ Add to specific playlist

                playlist_id = "PLy_LBg9TCEx_mQSJZXOLJDZ32VcRwJMU7"  # mrsclara notsoreal
                playlist_id = "PLy_LBg9TCEx-xrjfPoC0t7nEpnfrC5Me-"  #health and fitness
                
                playlist_id = "PLy_LBg9TCEx-bYra52rqWLNWNuIGDWTzE"  #horror
                playlist_id="PLy_LBg9TCEx9AcivqajhUmLoZyex2IXht" #kidsstories
                playlist_id = "PLy_LBg9TCEx9v0AginRCne_X_X8acGuzu" #finance
                try:
                    youtube.playlistItems().insert(
                        part="snippet",
                        body={
                            "snippet": {
                                "playlistId": playlist_id,
                                "resourceId": {
                                    "kind": "youtube#video",
                                    "videoId": video_id
                                }
                            }
                        }
                    ).execute()
                    logger.info(f"📂 Video added to playlist: {playlist_id}")
                except Exception as e:
                    logger.warning(f"❌ Failed to add video to playlist: {e}")

                return True
            else:
                logger.error("❌ Upload failed: No video ID returned.")
                return False

        except HttpError as e:
            logger.warning(f"⚠️ HTTP Error {e.resp.status}: {e.content}")
            if e.resp.status == 400:
                logger.error("❌ Bad Request. Check your metadata (title, description, tags, categoryId).")
                return False
            if e.resp.status not in [500, 502, 503, 504]:
                return False

        except (socket.timeout, TimeoutError, requests.exceptions.ReadTimeout) as e:
            logger.warning(f"⏱️ Read timeout: {e}")

        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning(f"🔌 Connection error: {e}")

        except Exception as e:
            logger.error(f"❗ Unexpected error: {e}")
            return False

        retry += 1
        if retry > max_retries:
            logger.error("❌ Upload failed after maximum retries.")
            return False

        sleep_time = min(60, 2 ** retry)
        logger.info(f"🔁 Retrying in {sleep_time:.2f} seconds (attempt {retry}/{max_retries})...")
        time.sleep(sleep_time)


if __name__ == "__main__":


    import os
    import time

    mapping_file = '/Users/uday/Downloads/VIDEOYT/ile_topic_map_song.txt'
  
    while True:
        # status = get_upload_status()
        # count = status["count"]

        # if count <= 0:
        #     print("✅ Upload limit reached for today.")
        #     break

        mappings = load_video_topic_mappings(mapping_file)

        if not mappings:
            print("⚠️ Mapping file is empty. Sleeping for 1 minute...")
            time.sleep(60)
            continue

        for video_path, topic in mappings:
            # if count <= 0:
            #     print("✅ Daily upload quota reached. Stopping.")
            #     break

            youtube, token_path = get_available_token()
            if not youtube:
                print("❌ All tokens failed or have reached the limit.")
                break

            if not os.path.exists(video_path):
                print(f"❌ Video file not found: {video_path}")
                continue  # Don't sleep, just skip to next video

            print(f"\n📤 Uploading Video: {video_path}")
            print(f"📝 Topic: {topic}")

            success = upload_video(video_path, topic)

            if success:
                try:
                    os.remove(video_path)
                    remove_video_entry(video_path)
                    remove_mapping_entry(video_path, mapping_file)
                    print(f"🗑️ Deleted: {video_path}")
                    # count -= 1
                    
                    # save_upload_status(count)
                    # print(f"✅ Upload complete. Remaining uploads today: {count}")
                except Exception as e:
                    print(f"⚠️ Failed to delete {video_path}: {e}")

        

        # print("⏳ Sleeping 1 minute before checking mapping file again...")
        # time.sleep(60)