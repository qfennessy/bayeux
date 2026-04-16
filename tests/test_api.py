"""End-to-end tests for the Nunchaku API.

These tests hit the live API — requires NUNCHAKU_API_KEY env var.

    # Run all tests:
    NUNCHAKU_API_KEY=sk-nunchaku-... pytest tests/test_api.py -v

    # Error tests only (no key needed):
    NUNCHAKU_API_KEY=fake pytest tests/test_api.py::TestErrors -v

Tests retry on 429 (rate limit) with the Retry-After header.
"""

import base64
import io
import os
import sys
import time

import pytest
import requests
from PIL import Image

# Allow importing the client wrapper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
from nunchaku import NunchakuClient

API_KEY = os.environ.get("NUNCHAKU_API_KEY", "")
BASE_URL = "https://api.nunchaku.dev"

pytestmark = pytest.mark.skipif(not API_KEY, reason="NUNCHAKU_API_KEY not set")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_RETRIES = 12  # 12 * 10s = 2 min max wait for rate limit
RETRY_DELAY = 10  # seconds between retries on 429


def api_post(path: str, payload: dict, timeout: int = 300) -> requests.Response:
    """POST with automatic retry on 429 (rate limit / concurrent limit)."""
    url = f"{BASE_URL}{path}"
    # Accept-Encoding: identity avoids brotli decompression issues with large video responses
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "Accept-Encoding": "identity"}
    for attempt in range(MAX_RETRIES):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 429:
            return resp
        wait = int(resp.headers.get("Retry-After", RETRY_DELAY))
        print(f"  429 rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
        time.sleep(wait)
    return resp  # return last 429 if exhausted


def make_test_image(width: int = 256, height: int = 256) -> bytes:
    """Create a simple test image (solid color with a circle)."""
    img = Image.new("RGB", (width, height), (135, 206, 235))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    cx, cy = width // 2, height // 2
    r = min(width, height) // 4
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def assert_b64_image(b64_str: str):
    """Assert that a base64 string decodes to a valid image."""
    raw = base64.b64decode(b64_str)
    assert len(raw) > 1000, f"Image too small: {len(raw)} bytes"
    img = Image.open(io.BytesIO(raw))
    assert img.size[0] > 0 and img.size[1] > 0


def assert_b64_video(b64_str: str):
    """Assert that a base64 string decodes to a non-trivial MP4."""
    raw = base64.b64decode(b64_str)
    assert len(raw) > 10_000, f"Video too small: {len(raw)} bytes"


# ---------------------------------------------------------------------------
# Text-to-Image
# ---------------------------------------------------------------------------


class TestTextToImage:
    """POST /v1/images/generations"""

    def test_basic(self):
        resp = api_post(
            "/v1/images/generations",
            {
                "model": "nunchaku-qwen-image",
                "prompt": "a red circle on white background",
                "n": 1,
                "size": "512x512",
                "tier": "radically_fast",
                "response_format": "b64_json",
                "seed": 42,
            },
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) == 1
        assert "created" in data
        assert isinstance(data["created"], int)
        assert_b64_image(data["data"][0]["b64_json"])

    def test_seed_produces_image(self):
        """Seed parameter is accepted and produces a valid image."""
        resp = api_post(
            "/v1/images/generations",
            {
                "model": "nunchaku-qwen-image",
                "prompt": "a blue square",
                "n": 1,
                "size": "512x512",
                "tier": "radically_fast",
                "response_format": "b64_json",
                "seed": 12345,
            },
        )
        assert resp.status_code == 200, f"Failed: {resp.status_code}: {resp.text[:200]}"
        assert_b64_image(resp.json()["data"][0]["b64_json"])

    def test_flux_model(self):
        """FLUX.2 Klein 9B text-to-image (fast tier)."""
        resp = api_post(
            "/v1/images/generations",
            {
                "model": "nunchaku-flux.2-klein-9b",
                "prompt": "a red triangle",
                "n": 1,
                "size": "512x512",
                "tier": "fast",
                "response_format": "b64_json",
            },
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        assert_b64_image(resp.json()["data"][0]["b64_json"])


# ---------------------------------------------------------------------------
# Image-to-Image
# ---------------------------------------------------------------------------


class TestImageToImage:
    """POST /v1/images/edits"""

    def test_basic(self):
        test_img = make_test_image()
        img_b64 = base64.b64encode(test_img).decode()

        resp = api_post(
            "/v1/images/edits",
            {
                "model": "nunchaku-qwen-image-edit",
                "prompt": "make the circle green",
                "url": f"data:image/jpeg;base64,{img_b64}",
                "n": 1,
                "size": "512x512",
                "tier": "radically_fast",
                "response_format": "b64_json",
            },
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        assert len(data["data"]) == 1
        assert_b64_image(data["data"][0]["b64_json"])

    def test_flux_model(self):
        """FLUX.2 Klein 9B image-to-image (fast tier)."""
        test_img = make_test_image()
        img_b64 = base64.b64encode(test_img).decode()

        resp = api_post(
            "/v1/images/edits",
            {
                "model": "nunchaku-flux.2-klein-9b-edit",
                "prompt": "make it look like a painting",
                "url": f"data:image/jpeg;base64,{img_b64}",
                "n": 1,
                "size": "512x512",
                "tier": "fast",
                "response_format": "b64_json",
            },
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        assert_b64_image(resp.json()["data"][0]["b64_json"])


# ---------------------------------------------------------------------------
# Text-to-Video
# ---------------------------------------------------------------------------


class TestTextToVideo:
    """POST /v1/video/generations"""

    def test_basic(self):
        resp = api_post(
            "/v1/video/generations",
            {
                "model": "nunchaku-wan2.2-lightning-t2v",
                "prompt": "a red ball bouncing",
                "n": 1,
                "size": "1280x720",
                "num_frames": 81,
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
                "response_format": "b64_json",
            },
            timeout=120,
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        assert len(data["data"]) == 1
        assert_b64_video(data["data"][0]["b64_json"])


# ---------------------------------------------------------------------------
# Image-to-Video
# ---------------------------------------------------------------------------


class TestImageToVideo:
    """POST /v1/video/animations"""

    def test_basic(self):
        test_img = make_test_image(1280, 720)
        img_b64 = base64.b64encode(test_img).decode()
        data_uri = f"data:image/jpeg;base64,{img_b64}"
        prompt = "the red ball starts bouncing"

        resp = api_post(
            "/v1/video/animations",
            {
                "model": "nunchaku-wan2.2-lightning-i2v",
                "prompt": prompt,
                "n": 1,
                "size": "1280x720",
                "num_frames": 81,
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
                "response_format": "b64_json",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            },
            timeout=120,
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        assert len(data["data"]) == 1
        assert_b64_video(data["data"][0]["b64_json"])


# ---------------------------------------------------------------------------
# Client wrapper
# ---------------------------------------------------------------------------


class TestNunchakuClient:
    """Test the demo/nunchaku.py client wrapper."""

    def test_text_to_image(self):
        client = NunchakuClient()
        result = client.text_to_image(
            "a red circle", size="512x512", tier="radically_fast", seed=42
        )
        assert isinstance(result, bytes)
        assert len(result) > 1000
        img = Image.open(io.BytesIO(result))
        assert img.size[0] > 0

    def test_edit_image(self):
        client = NunchakuClient()
        test_img = make_test_image()
        result = client.edit_image(
            test_img, "make it green", size="512x512", tier="radically_fast"
        )
        assert isinstance(result, bytes)
        assert len(result) > 1000

    def test_text_to_video(self):
        client = NunchakuClient()
        result = client.text_to_video("a ball bouncing")
        assert isinstance(result, bytes)
        assert len(result) > 10_000

    def test_image_to_video(self):
        client = NunchakuClient()
        test_img = make_test_image(1280, 720)
        result = client.image_to_video(test_img, "the ball starts bouncing")
        assert isinstance(result, bytes)
        assert len(result) > 10_000


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    """Verify error responses are well-formed."""

    def test_missing_key(self):
        resp = requests.post(
            f"{BASE_URL}/v1/images/generations",
            headers={"Content-Type": "application/json"},
            json={
                "model": "nunchaku-qwen-image",
                "prompt": "test",
                "n": 1,
            },
        )
        assert resp.status_code == 401

    def test_invalid_key(self):
        resp = requests.post(
            f"{BASE_URL}/v1/images/generations",
            headers={"Authorization": "Bearer sk-nunchaku-invalid", "Content-Type": "application/json"},
            json={
                "model": "nunchaku-qwen-image",
                "prompt": "test",
                "n": 1,
            },
        )
        assert resp.status_code == 401
