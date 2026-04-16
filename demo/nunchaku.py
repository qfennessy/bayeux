"""Nunchaku API client — thin wrapper around all 4 generation endpoints.

Usage:
    from nunchaku import NunchakuClient

    client = NunchakuClient()  # reads NUNCHAKU_API_KEY from env
    image_bytes = client.text_to_image("a red apple on a wooden table")
"""

import base64
import io
import os
import time
from pathlib import Path

import requests

BASE_URL = "https://api.nunchaku.dev"
MAX_RETRIES = 12
RETRY_DELAY = 10


class NunchakuClient:
    """Minimal client for the Nunchaku image/video generation API."""

    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL):
        self.api_key = api_key or os.environ.get("NUNCHAKU_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        if not self.api_key:
            raise ValueError(
                "API key required. Pass api_key= or set NUNCHAKU_API_KEY env var."
            )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }

    def _post(self, path: str, payload: dict, timeout: int = 300) -> dict:
        for attempt in range(MAX_RETRIES):
            resp = requests.post(
                f"{self.base_url}{path}",
                headers=self._headers(),
                json=payload,
                timeout=timeout,
            )
            if resp.status_code != 429:
                break
            wait = int(resp.headers.get("Retry-After", RETRY_DELAY))
            time.sleep(wait)
        if not resp.ok:
            raise RuntimeError(f"API error {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    # -- Text-to-Image --------------------------------------------------------

    def text_to_image(
        self,
        prompt: str,
        model: str = "nunchaku-qwen-image",
        size: str = "1024x1024",
        tier: str = "fast",
        seed: int | None = None,
        negative_prompt: str | None = None,
        **kwargs,
    ) -> bytes:
        """Generate an image from a text prompt. Returns raw image bytes (JPEG)."""
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "tier": tier,
            "response_format": "b64_json",
            **kwargs,
        }
        if seed is not None:
            payload["seed"] = seed
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        data = self._post("/v1/images/generations", payload)
        return base64.b64decode(data["data"][0]["b64_json"])

    # -- Image-to-Image (Edit) ------------------------------------------------

    def edit_image(
        self,
        image: str | bytes | Path,
        prompt: str,
        model: str = "nunchaku-qwen-image-edit",
        size: str = "1024x1024",
        tier: str = "fast",
        seed: int | None = None,
        **kwargs,
    ) -> bytes:
        """Edit an image with a text prompt. Returns raw image bytes (JPEG).

        `image` can be a file path, raw bytes, or a base64 string.
        """
        image_b64 = self._to_base64(image)
        mime = "image/png" if isinstance(image, (str, Path)) and str(image).endswith(".png") else "image/jpeg"
        data_uri = f"data:{mime};base64,{image_b64}"

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "url": data_uri,
            "n": 1,
            "size": size,
            "tier": tier,
            "response_format": "b64_json",
            **kwargs,
        }
        if seed is not None:
            payload["seed"] = seed

        data = self._post("/v1/images/edits", payload)
        return base64.b64decode(data["data"][0]["b64_json"])

    # -- Text-to-Video ---------------------------------------------------------

    def text_to_video(
        self,
        prompt: str,
        model: str = "nunchaku-wan2.2-lightning-t2v",
        size: str = "1280x720",
        num_frames: int = 81,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        seed: int | None = None,
        **kwargs,
    ) -> bytes:
        """Generate a video from a text prompt. Returns raw MP4 bytes."""
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "response_format": "b64_json",
            **kwargs,
        }
        if seed is not None:
            payload["seed"] = seed

        data = self._post("/v1/video/generations", payload)
        return base64.b64decode(data["data"][0]["b64_json"])

    # -- Image-to-Video --------------------------------------------------------

    def image_to_video(
        self,
        image: str | bytes | Path,
        prompt: str,
        model: str = "nunchaku-wan2.2-lightning-i2v",
        size: str = "1280x720",
        num_frames: int = 81,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        seed: int | None = None,
        **kwargs,
    ) -> bytes:
        """Animate an image into a video. Returns raw MP4 bytes.

        `image` can be a file path, raw bytes, or a base64 string.
        """
        image_b64 = self._to_base64(image)
        mime = "image/png" if isinstance(image, (str, Path)) and str(image).endswith(".png") else "image/jpeg"
        data_uri = f"data:{mime};base64,{image_b64}"

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
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
            **kwargs,
        }
        if seed is not None:
            payload["seed"] = seed

        data = self._post("/v1/video/animations", payload)
        return base64.b64decode(data["data"][0]["b64_json"])

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _to_base64(image: str | bytes | Path) -> str:
        """Convert a file path, raw bytes, or base64 string to base64."""
        if isinstance(image, (str, Path)) and Path(image).is_file():
            return base64.b64encode(Path(image).read_bytes()).decode()
        if isinstance(image, bytes):
            return base64.b64encode(image).decode()
        # Assume it's already a base64 string
        return image

    @staticmethod
    def save(data: bytes, path: str) -> str:
        """Write bytes to a file and return the path."""
        Path(path).write_bytes(data)
        return path
