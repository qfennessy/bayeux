"""Text-to-Image — generate an image from a text prompt.

Endpoint: POST /v1/images/generations
Model:    nunchaku-qwen-image (supports tiers: fast, radically_fast)

Usage:
    export NUNCHAKU_API_KEY="sk-nunchaku-..."
    python text_to_image.py
"""

import base64
import os

import requests

API_KEY = os.environ["NUNCHAKU_API_KEY"]
BASE_URL = "https://api.nunchaku.dev"

response = requests.post(
    f"{BASE_URL}/v1/images/generations",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    timeout=180,
    json={
        "model": "nunchaku-qwen-image",
        # nunchaku-qwen-image renders legible in-image text when the prompt
        # (a) quotes the literal string, (b) names a surface to put it on,
        # and (c) specifies a visual style. Meta-instructions like
        # "label this Cat" are unreliable.
        "prompt": (
            "a cat wearing sunglasses on a wooden table, photorealistic, "
            "studio lighting. A small wooden plaque propped on the table "
            "in front of the cat reads \"CAT\" in bold black serif letters."
        ),
        "n": 1,
        "size": "1024x1024",
        "tier": "fast",
        "num_inference_steps": 28,
        "response_format": "b64_json",
        "seed": 4242,
    },
)
response.raise_for_status()

img_bytes = base64.b64decode(response.json()["data"][0]["b64_json"])
with open("output_t2i.jpg", "wb") as f:
    f.write(img_bytes)
print(f"Saved output_t2i.jpg ({len(img_bytes):,} bytes)")
