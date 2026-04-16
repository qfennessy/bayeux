"""Using the Nunchaku API with the OpenAI Python SDK.

The image generation endpoints are OpenAI-compatible. Just swap the base_url.
Non-standard parameters (tier, num_frames, etc.) go in `extra_body`.

Install: pip install openai

Usage:
    export NUNCHAKU_API_KEY="sk-nunchaku-..."
    python openai_sdk.py
"""

import base64
import os

from openai import OpenAI

client = OpenAI(
    base_url="https://api.nunchaku.dev/v1",
    api_key=os.environ["NUNCHAKU_API_KEY"],
)

# --- Text-to-Image -----------------------------------------------------------

print("Generating image...")
response = client.images.generate(
    model="nunchaku-qwen-image",
    prompt="a red apple on a wooden table, photorealistic, studio lighting",
    n=1,
    size="1024x1024",
    response_format="b64_json",
    extra_body={"tier": "fast", "num_inference_steps": 28, "seed": 42},
)

img_bytes = base64.b64decode(response.data[0].b64_json)
with open("output_openai_t2i.jpg", "wb") as f:
    f.write(img_bytes)
print(f"Saved output_openai_t2i.jpg ({len(img_bytes):,} bytes)")

# --- Image Edit (if you have an input image) ---------------------------------

# Uncomment to try image editing:
#
# with open("input.jpg", "rb") as f:
#     img_b64 = base64.b64encode(f.read()).decode()
#
# response = client.images.edit(
#     model="nunchaku-qwen-image-edit",
#     image="unused",  # OpenAI SDK requires this but Nunchaku uses `url`
#     prompt="make it look like a watercolor painting",
#     n=1,
#     size="1024x1024",
#     response_format="b64_json",
#     extra_body={
#         "url": f"data:image/jpeg;base64,{img_b64}",
#         "tier": "fast",
#     },
# )
# edited = base64.b64decode(response.data[0].b64_json)
# with open("output_openai_i2i.jpg", "wb") as f:
#     f.write(edited)

print("Done!")
