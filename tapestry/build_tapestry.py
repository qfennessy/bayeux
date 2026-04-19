"""Build a Bayeux-style tapestry from a JSON of family stories.

Pipeline per story paragraph:
    1. Gemini 3.1 Flash converts the paragraph into a concise visual prompt.
    2. Nunchaku text-to-image (nunchaku-qwen-image) renders the prompt.
    3. All images are pasted into a cols x rows grid and written to disk.

A stable base seed is read from the input JSON; per-image seeds are derived
deterministically as base_seed + story_index so reruns are reproducible and
each panel is distinct.

Usage:
    export NUNCHAKU_API_KEY="sk-nunchaku-..."
    export GEMINI_API_KEY="..."
    python build_tapestry.py stories.json --out tapestry.jpg
"""

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import requests
from google import genai
from google.genai import errors as genai_errors
from PIL import Image, ImageDraw, ImageFilter, ImageFont


class _DropNonTextPartsWarning(logging.Filter):
    """Drop the ``google.genai.types`` warning about non-text parts.

    Gemini 3.x Flash models always return an empty ``thought_signature``
    part alongside text — even with ``thinking_budget=0`` — so the SDK's
    ``.text`` accessor logs a warning on every call. The text extraction
    is still correct, so the message is just noise for our use case.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "there are non-text parts in the response" not in record.getMessage()


logging.getLogger("google_genai.types").addFilter(_DropNonTextPartsWarning())

NUNCHAKU_URL = "https://api.nunchaku.dev/v1/images/generations"
NUNCHAKU_MODEL = "nunchaku-qwen-image"
NUNCHAKU_EDIT_URL = "https://api.nunchaku.dev/v1/images/edits"
NUNCHAKU_EDIT_MODEL = "nunchaku-qwen-image-edit"
SEAM_PATCH_SIZE = 1024  # image-edit model's fixed working size
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_EDIT_MODEL = "gemini-3.1-flash-image-preview"

PROMPT_SYSTEM_INSTRUCTION = (
    "You receive a JSON object with three fields: 'paragraph' (a passage "
    "of family history), 'people' (an ordered list of characters or "
    "roles present in that scene), and 'year' (an optional string like "
    "'1880' or '' when unknown). Return a JSON object with two fields: \n"
    "  1. 'title': a short narrative title for the scene, 3-6 words, "
    "Title Case, evocative and specific. When the 'year' field is "
    "non-empty, PREFIX the title with that year followed by a single "
    "space, for example: '1902 A Contract Under the Neem' or '1906 "
    "Eleanor at City Hall'. When 'year' is empty or missing, omit the "
    "year prefix — do not invent a year from the paragraph text. When "
    "the 'people' list is non-empty, reference at least one named "
    "character (or their relationship, like 'the Ricci Daughters') in "
    "the title where it reads naturally. No trailing punctuation. \n"
    "  2. 'prompt': a single concise visual image prompt describing ONE "
    "scene. When 'people' is non-empty, refer to each listed character "
    "by name or role at their first mention in the prompt, and give "
    "each a short physical descriptor appropriate to their role and era "
    "(age bracket, clothing, posture) so a text-to-image model can "
    "render a recognisable figure. Include concrete visual elements: "
    "who is present, what they are doing, the setting, key objects, "
    "time of day, and mood. Roughly 50-80 words. These are fictional "
    "family members, not public figures — naming them is fine. Do not "
    "include camera or lens jargon. Do not add preamble or labels. \n"
    "Return only the JSON object."
)

PROMPT_INSTRUCTION_HASH = hashlib.sha256(
    PROMPT_SYSTEM_INSTRUCTION.encode("utf-8")
).hexdigest()[:10]

GEMINI_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "prompt": {"type": "string"},
    },
    "required": ["title", "prompt"],
}


def gemini_title_and_prompt(
    client: genai.Client,
    paragraph: str,
    people: list[str] | None = None,
    year: str | None = None,
) -> tuple[str, str]:
    """Ask Gemini 3.1 Flash to return both a narrative title and a visual
    prompt for a paragraph. Returns ``(title, prompt)``.

    When ``people`` is non-empty, each character or role in the list gets
    a short visual descriptor in the generated prompt so the same person
    can be rendered consistently across multiple panels. When ``year`` is
    provided, the title is prefixed with that year (e.g. ``"1902 A
    Contract Under the Neem"``) — the model does not attempt to infer
    the year from the paragraph text. Uses structured JSON input AND
    output. Thinking is disabled.
    """
    request = {
        "paragraph": paragraph,
        "people": list(people) if people else [],
        "year": str(year).strip() if year else "",
    }
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=json.dumps(request, ensure_ascii=False),
        config={
            "system_instruction": PROMPT_SYSTEM_INSTRUCTION,
            "response_mime_type": "application/json",
            "response_schema": GEMINI_JSON_SCHEMA,
            "thinking_config": {"thinking_budget": 0},
        },
    )
    data = json.loads(resp.text)
    return data["title"].strip(), data["prompt"].strip()


_RATE_LIMIT_HEADER_PREFIXES = ("ratelimit", "x-ratelimit", "x-rate-limit")
_DIAGNOSTIC_HEADERS = (
    "retry-after",
    "cf-ray",
    "x-request-id",
    "x-amzn-requestid",
    "x-cloud-trace-context",
    "server",
    "date",
    "content-type",
)


def _describe_response(resp: "requests.Response") -> str:
    """Build a verbose multi-line description of an HTTP response.

    Includes status, request URL, rate-limit / diagnostic headers, and the
    response body (parsed as JSON when possible, otherwise truncated text).
    """
    lines = [f"status: {resp.status_code} {resp.reason}"]
    lines.append(f"url:    {resp.request.method} {resp.url}")
    interesting = {}
    for k, v in resp.headers.items():
        lk = k.lower()
        if lk in _DIAGNOSTIC_HEADERS or any(
            lk.startswith(p) for p in _RATE_LIMIT_HEADER_PREFIXES
        ):
            interesting[k] = v
    if interesting:
        lines.append("headers:")
        for k, v in interesting.items():
            lines.append(f"  {k}: {v}")
    body_text = resp.text or ""
    try:
        parsed = resp.json()
        body_render = json.dumps(parsed, indent=2, ensure_ascii=False)
    except (ValueError, json.JSONDecodeError):
        body_render = body_text
    if body_render:
        truncated = body_render if len(body_render) <= 2000 else body_render[:2000] + " …[truncated]"
        lines.append("body:")
        for line in truncated.splitlines():
            lines.append(f"  {line}")
    return "\n".join(lines)


def _describe_exception(e: Exception) -> str:
    """Verbose description for any exception raised during a request."""
    resp = getattr(e, "response", None)
    if resp is not None:
        return f"{type(e).__name__}: {e}\n{_describe_response(resp)}"
    return f"{type(e).__name__}: {e}"


def nunchaku_image(api_key: str, prompt: str, size: str, seed: int) -> bytes:
    """Render a prompt with the Nunchaku text-to-image endpoint.

    Retries on 429 and transient 5xx / connection errors. On every failed
    attempt the full response (status, diagnostic headers, parsed body) is
    printed so rate-limit or validation issues are visible without needing
    to re-run with a debugger. 429s honor the server's ``Retry-After`` header
    when provided, and otherwise back off more aggressively than 5xx since
    they reflect a rate-limit window, not a transient failure.
    """
    payload = {
        "model": NUNCHAKU_MODEL,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "tier": "radically_fast",
        "num_inference_steps": 28,
        "response_format": "b64_json",
        "seed": seed,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    max_attempts = 8
    last_err: Exception | None = None
    last_resp: "requests.Response | None" = None
    for attempt in range(max_attempts):
        try:
            resp = requests.post(
                NUNCHAKU_URL, headers=headers, json=payload, timeout=240
            )
            last_resp = resp
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(
                    f"{resp.status_code} from nunchaku", response=resp
                )
            if not resp.ok:
                # 4xx other than 429 — not retried, surface full detail
                print(
                    f"    nunchaku non-retriable error:\n{_describe_response(resp)}",
                    flush=True,
                )
                resp.raise_for_status()
            return base64.b64decode(resp.json()["data"][0]["b64_json"])
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)
            retry_after = None
            if status == 429 and resp is not None:
                ra = resp.headers.get("Retry-After")
                if ra and ra.isdigit():
                    retry_after = int(ra)
            if retry_after is not None:
                sleep_s = retry_after + random.random()
            elif status == 429:
                sleep_s = 20 * (attempt + 1) + random.random() * 5
            else:
                sleep_s = 2 ** attempt + random.random()

            print(
                f"    attempt {attempt + 1}/{max_attempts} failed — "
                f"sleeping {sleep_s:.1f}s then retrying",
                flush=True,
            )
            detail = _describe_exception(e)
            for line in detail.splitlines():
                print(f"      {line}", flush=True)
            time.sleep(sleep_s)
    # Exhausted retries — re-raise with a verbose final description.
    tail = (
        "\n" + _describe_response(last_resp) if last_resp is not None else ""
    )
    raise RuntimeError(
        f"nunchaku failed after {max_attempts} attempts: {last_err}{tail}"
    )


def nunchaku_edit(
    api_key: str, patch: Image.Image, prompt: str
) -> Image.Image:
    """Send a PIL image to the image-edit endpoint and return the edited result.

    Uses the same retry / verbose-error policy as ``nunchaku_image``.
    """
    buf = io.BytesIO()
    patch.save(buf, format="JPEG", quality=92)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {
        "model": NUNCHAKU_EDIT_MODEL,
        "prompt": prompt,
        "url": f"data:image/jpeg;base64,{img_b64}",
        "n": 1,
        "size": f"{patch.width}x{patch.height}",
        "tier": "fast",
        # Higher step count than text-to-image: the model tends to be
        # conservative on edits at this tier, and extra steps make the
        # seam-removal actually take effect.
        "num_inference_steps": 40,
        "response_format": "b64_json",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    max_attempts = 8
    last_err: Exception | None = None
    last_resp: "requests.Response | None" = None
    for attempt in range(max_attempts):
        try:
            resp = requests.post(
                NUNCHAKU_EDIT_URL, headers=headers, json=payload, timeout=240
            )
            last_resp = resp
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(
                    f"{resp.status_code} from nunchaku-edit", response=resp
                )
            if not resp.ok:
                print(
                    f"    nunchaku-edit non-retriable error:\n{_describe_response(resp)}",
                    flush=True,
                )
                resp.raise_for_status()
            data = base64.b64decode(resp.json()["data"][0]["b64_json"])
            return Image.open(io.BytesIO(data)).convert("RGB")
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)
            retry_after = None
            if status == 429 and resp is not None:
                ra = resp.headers.get("Retry-After")
                if ra and ra.isdigit():
                    retry_after = int(ra)
            if retry_after is not None:
                sleep_s = retry_after + random.random()
            elif status == 429:
                sleep_s = 20 * (attempt + 1) + random.random() * 5
            else:
                sleep_s = 2 ** attempt + random.random()
            print(
                f"    edit attempt {attempt + 1}/{max_attempts} failed — "
                f"sleeping {sleep_s:.1f}s then retrying",
                flush=True,
            )
            detail = _describe_exception(e)
            for line in detail.splitlines():
                print(f"      {line}", flush=True)
            time.sleep(sleep_s)
    tail = (
        "\n" + _describe_response(last_resp) if last_resp is not None else ""
    )
    raise RuntimeError(
        f"nunchaku-edit failed after {max_attempts} attempts: {last_err}{tail}"
    )


def _gemini_edit_once(
    client: genai.Client, patch: Image.Image, prompt: str
) -> Image.Image:
    """One attempt at a Gemini image edit. Raises on API error or when
    the response carries no image part."""
    buf = io.BytesIO()
    patch.save(buf, format="JPEG", quality=92)
    resp = client.models.generate_content(
        model=GEMINI_EDIT_MODEL,
        contents=[
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(buf.getvalue()).decode(),
                }
            },
            prompt,
        ],
    )
    for part in resp.candidates[0].content.parts:
        data = getattr(part, "inline_data", None)
        if data is not None and getattr(data, "data", None):
            raw = data.data
            if isinstance(raw, str):
                raw = base64.b64decode(raw)
            edited = Image.open(io.BytesIO(raw)).convert("RGB")
            if edited.size != patch.size:
                edited = edited.resize(patch.size, Image.LANCZOS)
            return edited
    raise RuntimeError(
        f"gemini-edit returned no image parts. "
        f"candidate.content.parts: {resp.candidates[0].content.parts}"
    )


def gemini_edit_image(
    client: genai.Client, patch: Image.Image, prompt: str
) -> Image.Image:
    """Edit an image via Gemini 3.1 Flash Image Preview, with retries.

    Retries on Gemini 5xx (server overload, deadline expired) and 429
    (rate limit). The google-genai SDK has its own short internal
    retry, but its default policy gives up quickly on 503s from the
    image preview model — we wrap it to keep seam blending resilient.
    Non-retriable 4xx errors (400, 403, 404) propagate immediately so a
    model-name typo or quota-auth problem fails loudly.
    """
    max_attempts = 8
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return _gemini_edit_once(client, patch, prompt)
        except genai_errors.APIError as e:
            last_err = e
            code = getattr(e, "code", None)
            retriable = (
                code is None  # unknown — safer to retry once or twice
                or code == 429
                or (isinstance(code, int) and 500 <= code < 600)
            )
            if not retriable or attempt == max_attempts - 1:
                raise
            if code == 429:
                sleep_s = 20 * (attempt + 1) + random.random() * 5
            else:
                sleep_s = 2 ** attempt + random.random()
            print(
                f"    gemini-edit attempt {attempt + 1}/{max_attempts} "
                f"failed (code={code}) — sleeping {sleep_s:.1f}s: {e}",
                flush=True,
            )
            time.sleep(sleep_s)
        except RuntimeError as e:
            # "No image parts" can happen when the model safety-filtered
            # the request; retry once in case it's transient, then fail.
            last_err = e
            if attempt >= 1:
                raise
            sleep_s = 2.0 + random.random()
            print(
                f"    gemini-edit attempt {attempt + 1}/{max_attempts} "
                f"returned no image — retrying in {sleep_s:.1f}s: {e}",
                flush=True,
            )
            time.sleep(sleep_s)
    raise RuntimeError(
        f"gemini-edit failed after {max_attempts} attempts: {last_err}"
    )


def _draw_seam_indicator(
    patch: Image.Image, local_x: int, color=(255, 0, 255), width: int = 6
) -> Image.Image:
    """Return a copy of ``patch`` with a solid magenta vertical line
    drawn at column ``local_x``.

    Giving the edit model a concrete visual target (a bright,
    unmistakably-artificial line) tends to produce a stronger, more
    localized repaint than asking it to 'find the seam itself' — the
    model both erases the line and smooths the region around it in
    a single pass.
    """
    marked = patch.copy()
    draw = ImageDraw.Draw(marked)
    half = width // 2
    draw.rectangle(
        [(local_x - half, 0), (local_x + (width - half), marked.height)],
        fill=color,
    )
    return marked


def _seam_feather_mask(size: int, feather: int) -> Image.Image:
    """Soft rectangular mask: white in the center, fading to black near
    the edges over ``feather`` pixels. Used to blend an edited patch back
    into the tapestry without visible patch boundaries.
    """
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((feather, feather, size - feather, size - feather), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(feather / 2))


def _seam_patch_centers(
    cols: int, rows: int, tile_w: int, tile_h: int, vertical_only: bool = True
) -> list[tuple[int, int]]:
    """Compute the (x, y) center of each seam-segment patch.

    For every vertical seam at x = c*tile_w (c in 1..cols-1), one patch
    sits at the midpoint of each row. When ``vertical_only`` is False we
    additionally emit patches for every horizontal seam.

    Horizontal seams are skipped by default because the caption bar
    drawn by ``label_tile`` sits right at the bottom of each tile — a
    horizontal-seam patch runs straight through it and the edit model
    reliably corrupts the label text when asked to smooth that region.
    """
    centers: list[tuple[int, int]] = []
    for c in range(1, cols):
        x = c * tile_w
        for r in range(rows):
            centers.append((x, r * tile_h + tile_h // 2))
    if not vertical_only:
        for r in range(1, rows):
            y = r * tile_h
            for c in range(cols):
                centers.append((c * tile_w + tile_w // 2, y))
    return centers


def blend_seams(
    tapestry_path: Path,
    out_path: Path,
    cols: int,
    rows: int,
    tile_w: int,
    tile_h: int,
    edit_provider: str,
    nunchaku_key: str,
    gemini_client: genai.Client,
    patch_cache_dir: Path,
    verbose: bool = False,
) -> None:
    """Post-process a completed tapestry to smooth the seams where panels meet.

    Walks each vertical seam (horizontal seams are skipped by default —
    the caption bar lives there), extracts 1024x1024 patches straddling
    the seam, runs each patch through the configured image-edit backend
    with a seam-blending prompt, then pastes the edited patch back
    through a feathered mask so patch boundaries dissolve into the rest
    of the tapestry.

    ``edit_provider`` selects the backend:
        ``"nunchaku"`` → ``nunchaku-qwen-image-edit`` (fast, cheap,
          conservative — can fail to actually remove the seam).
        ``"gemini"``   → ``gemini-3.1-flash-image-preview`` (better
          instruction-following for "preserve content, only fix seam";
          somewhat slower).
    """
    if edit_provider not in ("nunchaku", "gemini"):
        raise ValueError(f"unknown edit_provider: {edit_provider!r}")
    img = Image.open(tapestry_path).convert("RGB")
    W, H = img.size
    patch = SEAM_PATCH_SIZE
    if tile_w < patch // 2 or tile_h < patch // 2:
        print(
            f"warning: tiles are {tile_w}x{tile_h}, smaller than half the "
            f"{patch}x{patch} edit window. Seam blending will still run, but "
            "patches will span several tiles and the edit may smear content.",
            file=sys.stderr,
            flush=True,
        )

    centers = _seam_patch_centers(cols, rows, tile_w, tile_h)
    feather = patch // 4  # 256px for a 1024 patch
    mask = _seam_feather_mask(patch, feather)
    patch_cache_dir.mkdir(parents=True, exist_ok=True)

    # Per-provider prompt. Gemini gets a magenta-line indicator painted
    # on the seam (the model is strong enough to both erase the line
    # *and* blend the region around it in one pass). Nunchaku can't
    # reliably remove the line — it leaves visible magenta residue — so
    # we hand it a clean crop and a more traditional 'blend the seam'
    # instruction instead.
    use_indicator = edit_provider == "gemini"
    if use_indicator:
        prompt = (
            "This image shows two separately-drawn illustrations joined "
            "by a single VERTICAL seam running top-to-bottom. I have "
            "painted a bright MAGENTA vertical line directly on the "
            "image to mark exactly where that seam is. \n"
            "\n"
            "Your job: REMOVE the magenta line, AND at the same time "
            "blend the scenes on its left and right sides into one "
            "continuous illustration. Repaint the column directly under "
            "the magenta line and a few dozen pixels to either side of "
            "it so the sky, horizon, ground, foliage, walls, and shadows "
            "flow naturally across the former seam. Match colors, tones, "
            "and lighting. Do not leave any trace of magenta. Do not "
            "replace the magenta line with a different straight vertical "
            "edge. \n"
            "\n"
            "CRITICAL exception: there may be one or two thin horizontal "
            "rows of white outlined caption text somewhere in the image "
            "(panel titles). Do NOT modify that text or the glyphs "
            "inside it — copy it through pixel-for-pixel. \n"
            "\n"
            "Keep every character, object, facial expression, and pose "
            "exactly where they are."
        )
    else:
        prompt = (
            "This image contains a single sharp VERTICAL seam where two "
            "separately-drawn illustrations meet side by side. There is "
            "an abrupt change in sky color, ground color, horizon line, "
            "or lighting along that vertical line. Your job: eliminate "
            "the vertical seam. Extend the sky, ground, horizon, foliage, "
            "walls, and shadows smoothly from left to right across the "
            "former seam so the two halves read as one continuous "
            "illustration drawn by a single artist. Match colors and "
            "tones across the boundary. Do not draw any vertical straight "
            "edge, frame, rectangle, or line coincident with the former "
            "seam. Do not modify any horizontal row of outlined caption "
            "text that may appear in the image — leave any text or "
            "labels strictly unchanged. Keep every character, object, "
            "facial expression, and pose exactly where they are."
        )

    # Version the patch cache by a hash of the prompt text — if we tweak
    # the prompt (e.g. loosening preservation), old cached patches no
    # longer reflect the current behavior, so we look them up in a fresh
    # subdir instead of silently serving stale results.
    prompt_tag = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
    patch_cache_subdir = patch_cache_dir / prompt_tag
    patch_cache_subdir.mkdir(parents=True, exist_ok=True)

    print(
        f"blend-seams: {len(centers)} patches "
        f"(each {patch}x{patch}, feather {feather}px, provider "
        f"{edit_provider}) on {tapestry_path}",
        flush=True,
    )
    print(
        f"  patch cache: {patch_cache_subdir} (prompt_tag={prompt_tag})",
        flush=True,
    )
    run_start = time.monotonic()
    for i, (cx, cy) in enumerate(centers, 1):
        x0 = max(0, min(W - patch, cx - patch // 2))
        y0 = max(0, min(H - patch, cy - patch // 2))
        crop = img.crop((x0, y0, x0 + patch, y0 + patch))
        # Cache filename encodes the crop origin so reruns skip any seam
        # patch already edited by this provider at this prompt version.
        # A partial run (e.g. one that 503'd halfway through) leaves
        # finished patches on disk; changing the prompt text writes into
        # a new subdir so old work is preserved for comparison.
        patch_cache_path = patch_cache_subdir / f"x{x0:05d}-y{y0:05d}.jpg"
        t0 = time.monotonic()
        if patch_cache_path.exists():
            print(
                f"  [{i}/{len(centers)}] seam patch at ({cx},{cy}) "
                f"crop=({x0},{y0}) — cached",
                flush=True,
            )
            edited = Image.open(patch_cache_path).convert("RGB")
        else:
            print(
                f"  [{i}/{len(centers)}] seam patch at ({cx},{cy}) "
                f"crop=({x0},{y0}) via {edit_provider}...",
                flush=True,
            )
            # For Gemini, paint a bright magenta guideline on the seam
            # column within the patch and let the model both erase it
            # and blend the region around it — a concrete visual target
            # forces a stronger-than-identity repaint. Nunchaku leaves
            # magenta residue on its output, so it sees the unmarked
            # crop and a traditional 'eliminate the seam' prompt. In
            # both cases ``cx - x0`` is the seam's x-coordinate in
            # patch-local space (accounts for clamping at image edges).
            if use_indicator:
                input_patch = _draw_seam_indicator(crop, cx - x0)
            else:
                input_patch = crop
            if edit_provider == "gemini":
                edited = gemini_edit_image(gemini_client, input_patch, prompt)
            else:
                edited = nunchaku_edit(nunchaku_key, input_patch, prompt)
            if edited.size != (patch, patch):
                edited = edited.resize((patch, patch), Image.LANCZOS)
            edited.save(patch_cache_path, quality=92)
        img.paste(edited, (x0, y0), mask)
        print(
            f"    ok ({time.monotonic() - t0:.2f}s)",
            flush=True,
        )
        if verbose:
            print(
                f"    covers seam segment; patch {patch}x{patch}, "
                f"feather {feather}px",
                flush=True,
            )

    img.save(out_path, quality=92)
    print(
        f"blend-seams: wrote {out_path} "
        f"(total {time.monotonic() - run_start:.0f}s)",
        flush=True,
    )


def render_panel(
    story: dict,
    index: int,
    base_seed: int,
    style_suffix: str,
    image_size: str,
    prompt_dir: Path,
    image_dir: Path,
    gemini_client: genai.Client,
    nunchaku_key: str,
    verbose: bool = False,
) -> tuple[Path, dict]:
    """Generate (or load from cache) a single panel image.

    Title + prompt are cached together at ``prompt_dir/<id>.json`` (style-
    independent — Gemini only sees the family paragraph). Legacy caches
    that still hold a bare ``<id>.txt`` prompt file are migrated
    transparently: the prompt is preserved and we call Gemini once to
    fill in the title. Images are cached under ``image_dir`` (per style),
    so multiple styles of the same family share the Gemini work.

    Returns (image_path, stats). Stats keys: ``prompt_seconds``,
    ``image_seconds`` (0.0 when served from cache), ``prompt_cached``,
    ``image_cached``, ``prompt_chars``, ``image_bytes``, ``seed``,
    ``title``.
    """
    sid = story["id"]
    cache_path = prompt_dir / f"{sid}.json"
    legacy_prompt_path = prompt_dir / f"{sid}.txt"
    image_path = image_dir / f"{sid}.jpg"
    stats: dict = {
        "prompt_seconds": 0.0,
        "image_seconds": 0.0,
        "prompt_cached": False,
        "image_cached": False,
        "prompt_chars": 0,
        "image_bytes": 0,
        "seed": base_seed + index,
        "title": "",
    }

    current_people = story.get("people") or []
    current_year = (str(story.get("year")).strip() if story.get("year") else "")

    title: str | None = None
    visual_prompt: str | None = None

    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        cached_people = cached.get("people") or []
        cached_year = cached.get("year") or ""
        cached_instruction = cached.get("instruction_hash")
        people_match = cached_people == current_people
        year_match = cached_year == current_year
        instruction_match = cached_instruction == PROMPT_INSTRUCTION_HASH
        if people_match and year_match and instruction_match:
            title = cached["title"]
            visual_prompt = cached["prompt"]
            stats["prompt_cached"] = True
            stats["prompt_chars"] = len(visual_prompt)
            print(
                f"  gemini: cached title={title!r}, "
                f"{len(visual_prompt)} prompt chars, people={len(current_people)}",
                flush=True,
            )
        else:
            reasons = []
            if not people_match:
                reasons.append(
                    f"people {len(cached_people)}→{len(current_people)}"
                )
            if not year_match:
                reasons.append(
                    f"year {cached_year or 'none'}→{current_year or 'none'}"
                )
            if not instruction_match:
                reasons.append(
                    f"system-instruction {cached_instruction or 'none'}"
                    f"→{PROMPT_INSTRUCTION_HASH}"
                )
            print(
                f"  gemini: cache invalidated ({', '.join(reasons)}); "
                "regenerating title + prompt",
                flush=True,
            )

    if visual_prompt is None and legacy_prompt_path.exists() and not current_people:
        # Legacy cache (prompt-only .txt file) migrates only when the
        # story has no people list — otherwise we want a fresh,
        # people-aware prompt rather than a salvage of the old one.
        visual_prompt = legacy_prompt_path.read_text().strip()
        t0 = time.monotonic()
        print(f"  gemini: migrating cache — fetching title only...", flush=True)
        title, _ = gemini_title_and_prompt(
            gemini_client, story["paragraph"], year=current_year
        )
        elapsed = time.monotonic() - t0
        cache_path.write_text(
            json.dumps(
                {
                    "title": title,
                    "prompt": visual_prompt,
                    "people": current_people,
                    "year": current_year,
                    "instruction_hash": PROMPT_INSTRUCTION_HASH,
                },
                ensure_ascii=False,
            )
        )
        legacy_prompt_path.unlink()
        stats["prompt_seconds"] = elapsed
        stats["prompt_chars"] = len(visual_prompt)
        print(
            f"  gemini: migrated ({elapsed:.2f}s, title={title!r})",
            flush=True,
        )

    if visual_prompt is None:
        t0 = time.monotonic()
        print(
            f"  gemini: generating title + prompt (people={len(current_people)})...",
            flush=True,
        )
        title, visual_prompt = gemini_title_and_prompt(
            gemini_client, story["paragraph"], current_people, current_year
        )
        elapsed = time.monotonic() - t0
        cache_path.write_text(
            json.dumps(
                {
                    "title": title,
                    "prompt": visual_prompt,
                    "people": current_people,
                    "year": current_year,
                    "instruction_hash": PROMPT_INSTRUCTION_HASH,
                },
                ensure_ascii=False,
            )
        )
        if legacy_prompt_path.exists():
            legacy_prompt_path.unlink()
        stats["prompt_seconds"] = elapsed
        stats["prompt_chars"] = len(visual_prompt)
        print(
            f"  gemini: ok ({elapsed:.2f}s, title={title!r}, "
            f"{len(visual_prompt)} prompt chars, model {GEMINI_MODEL})",
            flush=True,
        )
    stats["title"] = title
    if verbose:
        preview = visual_prompt if len(visual_prompt) <= 400 else visual_prompt[:400] + " …"
        print(f"    > {preview}", flush=True)

    full_prompt = f"{visual_prompt} {style_suffix}".strip()

    if image_path.exists():
        size_bytes = image_path.stat().st_size
        stats["image_cached"] = True
        stats["image_bytes"] = size_bytes
        print(f"  image:  cached ({size_bytes / 1024:.0f} KB)", flush=True)
    else:
        seed = stats["seed"]
        t0 = time.monotonic()
        if verbose:
            print(
                f"  image:  nunchaku seed={seed} size={image_size} "
                f"model={NUNCHAKU_MODEL} tier=radically_fast...",
                flush=True,
            )
        else:
            print(f"  image:  nunchaku seed={seed}...", flush=True)
        img_bytes = nunchaku_image(nunchaku_key, full_prompt, image_size, seed)
        elapsed = time.monotonic() - t0
        image_path.write_bytes(img_bytes)
        stats["image_seconds"] = elapsed
        stats["image_bytes"] = len(img_bytes)
        print(
            f"  image:  ok ({elapsed:.2f}s, {len(img_bytes) / 1024:.0f} KB)",
            flush=True,
        )

    return image_path, stats


_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)


def _load_font(size: int, explicit: Path | None = None) -> ImageFont.ImageFont:
    """Load a TrueType font at ``size`` px, trying common system paths.

    Falls back to PIL's bitmap default with a stderr warning when nothing
    suitable is found — the default is pixelated at 1024px tile scale, so
    real TTFs are strongly preferred.
    """
    paths = [str(explicit)] if explicit else list(_FONT_CANDIDATES)
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    print(
        "warning: no TrueType font found; labels will use the tiny PIL "
        "default. Pass --font PATH to override.",
        file=sys.stderr,
    )
    return ImageFont.load_default()


def label_tile(
    tile: Image.Image, text: str, font: ImageFont.ImageFont
) -> Image.Image:
    """Return a copy of ``tile`` with ``text`` drawn at the bottom.

    Style: white text with a dark outline stroke (no tinted band).
    Stroke width scales with font size so labels stay crisp from 512px
    thumbnails through 2048px hero renders without being comically thick
    at either end. Operates on a copy so callers can reuse the original
    (cached) image object.
    """
    labeled = tile.copy()
    draw = ImageDraw.Draw(labeled)
    w, h = labeled.size
    font_px = getattr(font, "size", None) or max(14, int(h * 0.045))
    stroke = max(2, int(font_px * 0.15))
    bottom_pad = max(8, int(h * 0.02))
    text_w = draw.textlength(text, font=font)
    text_y = h - font_px - bottom_pad - stroke
    draw.text(
        ((w - text_w) / 2, text_y),
        text,
        fill="white",
        font=font,
        stroke_width=stroke,
        stroke_fill="black",
    )
    return labeled


def frame_tile(tile: Image.Image, thickness: int) -> Image.Image:
    """Return a copy of ``tile`` with a thin decorative frame painted
    around its edges.

    The frame is drawn over the tile's outermost pixels (not added as
    extra canvas), so it does not change tile dimensions or grid math.
    Two-tone: an outer thin dark outline plus an inner cream inset
    that together evoke a panel-style illumination frame. Used to turn
    the tile boundary into a deliberate design element rather than a
    defect to hide.
    """
    if thickness <= 0:
        return tile
    framed = tile.copy()
    draw = ImageDraw.Draw(framed)
    w, h = framed.size
    dark = (38, 28, 18)
    cream = (238, 227, 200)
    outer = max(1, thickness // 3)
    # Dark outer band.
    for i in range(outer):
        draw.rectangle([(i, i), (w - 1 - i, h - 1 - i)], outline=dark)
    # Cream inset band.
    for i in range(outer, thickness):
        draw.rectangle([(i, i), (w - 1 - i, h - 1 - i)], outline=cream)
    return framed


def _horizontal_alpha_gradient(
    width: int, height: int, feather_px: int
) -> Image.Image:
    """Build an L-mode alpha mask the shape of a tile: a linear 0→255
    ramp over the first ``feather_px`` columns, then fully opaque. Used
    when pasting the Nth tile (N > 0) over the previous tile's right
    edge to produce a cross-dissolve along the seam.
    """
    import numpy as np

    arr = np.full((height, width), 255, dtype=np.uint8)
    if feather_px > 0:
        ramp = np.linspace(0, 255, feather_px, dtype=np.uint8)
        arr[:, :feather_px] = ramp[None, :]
    return Image.fromarray(arr, mode="L")


def assemble_tapestry(
    panels: list[Path],
    cols: int,
    rows: int,
    out_path: Path,
    labels: list[str] | None = None,
    font_path: Path | None = None,
    feather_px: int = 0,
    frame_px: int = 0,
) -> None:
    """Paste all panels into a cols x rows grid and save.

    When ``labels`` is provided (one string per panel), each tile gets
    its title drawn at the bottom with an outlined white stroke. Labels
    are only drawn on the final tapestry, never on the cached per-panel
    images, so you can change or disable them without regenerating.

    ``feather_px > 0`` overlaps adjacent columns by ``feather_px``
    pixels and cross-dissolves them with a horizontal alpha gradient.
    This narrows the output canvas by ``(cols - 1) * feather_px`` and
    produces a genuinely continuous transition at each vertical seam at
    the cost of both tiles showing faintly through each other in the
    overlap zone. Rows are never feathered — caption text lives there.

    ``frame_px > 0`` paints a thin two-tone frame around each tile
    before it is pasted, turning visible seams into deliberate panel
    borders.
    """
    if len(panels) != cols * rows:
        raise ValueError(
            f"expected {cols * rows} panels for {cols}x{rows} grid, got {len(panels)}"
        )
    if labels is not None and len(labels) != len(panels):
        raise ValueError(
            f"labels: {len(labels)} given, expected {len(panels)}"
        )

    tiles = [Image.open(p).convert("RGB") for p in panels]
    w, h = tiles[0].size
    feather_px = max(0, min(feather_px, w - 1))
    canvas_w = cols * w - (cols - 1) * feather_px
    canvas = Image.new("RGB", (canvas_w, rows * h), "white")
    font = _load_font(int(h * 0.045), font_path) if labels else None
    gradient_mask = (
        _horizontal_alpha_gradient(w, h, feather_px) if feather_px > 0 else None
    )
    for i, tile in enumerate(tiles):
        if tile.size != (w, h):
            tile = tile.resize((w, h), Image.LANCZOS)
        if frame_px > 0:
            tile = frame_tile(tile, frame_px)
        if labels:
            tile = label_tile(tile, labels[i], font)
        r, c = divmod(i, cols)
        x = c * (w - feather_px)
        y = r * h
        if feather_px > 0 and c > 0:
            canvas.paste(tile, (x, y), gradient_mask)
        else:
            canvas.paste(tile, (x, y))
    canvas.save(out_path, quality=92)


def poisson_blend_seams(
    tapestry_path: Path,
    out_path: Path,
    cols: int,
    rows: int,
    tile_w: int,
    strip_width: int = 192,
) -> None:
    """Gradient-domain (Poisson) smoothing of every vertical seam.

    Uses OpenCV's ``cv2.seamlessClone`` in ``MIXED_CLONE`` mode on a
    vertical strip of width ``2 * strip_width`` centered on each seam.
    MIXED_CLONE keeps the stronger gradient from either the surrounding
    tapestry or the strip, which preserves detail on both sides while
    matching the boundary colors — typically the seam becomes invisible
    on sky / ground / background-dominated boundaries. Writes the
    blended result to ``out_path`` alongside the raw tapestry.

    ``opencv-python`` is imported lazily so users who never pass
    ``--poisson`` don't need it installed.
    """
    import cv2
    import numpy as np

    bgr = cv2.imread(str(tapestry_path))
    if bgr is None:
        raise RuntimeError(f"could not read {tapestry_path}")
    h_img, w_img = bgr.shape[:2]
    result = bgr.copy()
    applied = 0
    for c in range(1, cols):
        seam_x = c * tile_w
        left = max(0, seam_x - strip_width)
        right = min(w_img, seam_x + strip_width)
        if right - left < 4:
            continue
        src = result[:, left:right].copy()
        mask = np.full((src.shape[0], src.shape[1]), 255, dtype=np.uint8)
        center = ((left + right) // 2, h_img // 2)
        result = cv2.seamlessClone(src, result, mask, center, cv2.MIXED_CLONE)
        applied += 1
    cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(
        f"poisson: wrote {out_path} "
        f"(blended {applied} vertical seams, strip ±{strip_width}px)",
        flush=True,
    )


def pick_grid(n: int) -> tuple[int, int]:
    """Pick a near-square (cols, rows) pair whose product is exactly n.

    For n with integer factors this gives the most-square factorization
    (cols >= rows). Primes fall back to a 1-row strip.
    """
    for rows in range(int(n**0.5), 0, -1):
        if n % rows == 0:
            return (n // rows, rows)
    return (n, 1)


def load_style(style_path: Path) -> tuple[str, str]:
    """Load a style JSON. Returns (style_name, style_suffix).

    The style name falls back to the file stem so the user never has to
    keep the JSON's ``name`` field in sync with the filename.
    """
    data = json.loads(style_path.read_text())
    suffix = data.get("style_suffix", "").strip()
    if not suffix:
        raise ValueError(f"{style_path}: missing non-empty 'style_suffix'")
    return data.get("name", style_path.stem), suffix


def build_one(
    stories_path: Path,
    out_path: Path,
    cache_root: Path,
    style_name: str,
    style_suffix: str,
    gemini_client: genai.Client,
    nunchaku_key: str,
    limit: int | None = None,
    verbose: bool = False,
    labels: bool = True,
    font_path: Path | None = None,
    blend: bool = False,
    edit_provider: str = "nunchaku",
    feather_px: int = 0,
    frame_px: int = 0,
    poisson: bool = False,
    regenerate_images: bool = False,
) -> None:
    """Render a single family into its own styled tapestry image.

    Family paragraphs come from ``stories_path``; the visual style is
    supplied separately via ``style_suffix`` (typically loaded from a
    style JSON under ``tapestry/styles/``). Prompts are cached per
    family and images are cached per (family, style).

    ``limit`` caps the number of stories used. When set, only the first
    ``limit`` entries are rendered and the grid is recomputed to fit them
    (ignoring the JSON's declared grid). When ``None``, all stories are
    used and the JSON's grid must match ``len(stories)``.
    """
    data = json.loads(stories_path.read_text())
    if "style_suffix" in data:
        raise ValueError(
            f"{stories_path}: family JSON should not contain 'style_suffix' "
            "— the style is now supplied via --style. Remove the field."
        )
    stories = data["stories"]
    base_seed = int(data["seed"])
    image_size = data.get("image_size", "1024x1024")

    if limit is not None and limit < len(stories):
        stories = stories[:limit]
        cols, rows = pick_grid(len(stories))
    else:
        cols = data["grid"]["cols"]
        rows = data["grid"]["rows"]
        if len(stories) != cols * rows:
            raise ValueError(
                f"{stories_path}: {len(stories)} stories but grid is "
                f"{cols}x{rows}={cols * rows}"
            )

    family_cache = cache_root / stories_path.stem
    prompt_dir = family_cache / "prompts"
    image_dir = family_cache / "images" / style_name
    if regenerate_images and image_dir.exists():
        # Nuke the per-(family, style) image dir, including any seam patches
        # built from the old images. Prompts are preserved so we keep the
        # people-aware Gemini work and only re-call Nunchaku for the panels.
        print(
            f"regenerate-images: clearing {image_dir} "
            f"(prompts under {prompt_dir} untouched)",
            flush=True,
        )
        shutil.rmtree(image_dir)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.monotonic()
    total = len(stories)
    print(
        f"=== {stories_path.name} × {style_name} ===\n"
        f"tapestry: {data.get('title', stories_path.stem)} | {total} panels "
        f"| grid {cols}x{rows} | prompts {prompt_dir} | images {image_dir} "
        f"| out {out_path}",
        flush=True,
    )
    panels: list[Path] = []
    all_stats: list[dict] = []
    for i, story in enumerate(stories):
        panel_start = time.monotonic()
        elapsed = time.monotonic() - run_start
        print(
            f"[{i + 1}/{total}] {story['id']} (t+{elapsed:.0f}s)",
            flush=True,
        )
        panel_path, stats = render_panel(
            story=story,
            index=i,
            base_seed=base_seed,
            style_suffix=style_suffix,
            image_size=image_size,
            prompt_dir=prompt_dir,
            image_dir=image_dir,
            gemini_client=gemini_client,
            nunchaku_key=nunchaku_key,
            verbose=verbose,
        )
        panels.append(panel_path)
        all_stats.append(stats)
        print(
            f"  panel done in {time.monotonic() - panel_start:.2f}s",
            flush=True,
        )

    total_gemini = sum(s["prompt_seconds"] for s in all_stats)
    total_nunchaku = sum(s["image_seconds"] for s in all_stats)
    gemini_calls = sum(1 for s in all_stats if not s["prompt_cached"])
    nunchaku_calls = sum(1 for s in all_stats if not s["image_cached"])
    print(
        f"summary: gemini {total_gemini:.2f}s across {gemini_calls} calls "
        f"({total_gemini / gemini_calls:.2f}s avg) | "
        f"nunchaku {total_nunchaku:.2f}s across {nunchaku_calls} calls "
        f"({total_nunchaku / nunchaku_calls:.2f}s avg) | "
        f"cached {total - gemini_calls} prompts, {total - nunchaku_calls} images"
        if gemini_calls and nunchaku_calls
        else f"summary: all {total} panels served from cache "
        f"(gemini {total_gemini:.2f}s, nunchaku {total_nunchaku:.2f}s)",
        flush=True,
    )
    if verbose:
        print("  per-panel timing:", flush=True)
        for i, s in enumerate(all_stats, 1):
            pc = "cache" if s["prompt_cached"] else f"{s['prompt_seconds']:.2f}s"
            ic = "cache" if s["image_cached"] else f"{s['image_seconds']:.2f}s"
            print(
                f"    {i:2d}. prompt {pc:>6}  image {ic:>7}  "
                f"({s['image_bytes'] / 1024:5.0f} KB, seed {s['seed']})",
                flush=True,
            )

    print(f"assembling {cols}x{rows} grid...", flush=True)
    if labels:
        # Priority: explicit per-story "label" > Gemini-generated narrative
        # title > "Image N" fallback. The explicit label lets a family
        # override what the LLM picked, without editing the cache.
        label_texts = [
            story.get("label")
            or all_stats[i]["title"]
            or f"Image {i + 1}"
            for i, story in enumerate(stories)
        ]
    else:
        label_texts = None
    assemble_tapestry(
        panels,
        cols,
        rows,
        out_path,
        labels=label_texts,
        font_path=font_path,
        feather_px=feather_px,
        frame_px=frame_px,
    )
    print(
        f"wrote {out_path} (total {time.monotonic() - run_start:.0f}s)",
        flush=True,
    )

    if blend:
        tile_w, tile_h = Image.open(panels[0]).size
        blended_path = out_path.with_name(f"{out_path.stem}-blended{out_path.suffix}")
        # Seam patches are style- and provider-specific: nunchaku and
        # gemini will edit the same crop differently, so cache them under
        # separate subdirs so switching providers doesn't reuse the
        # other's results.
        patch_cache_dir = image_dir / "seams" / edit_provider
        blend_seams(
            tapestry_path=out_path,
            out_path=blended_path,
            cols=cols,
            rows=rows,
            tile_w=tile_w,
            tile_h=tile_h,
            edit_provider=edit_provider,
            nunchaku_key=nunchaku_key,
            gemini_client=gemini_client,
            patch_cache_dir=patch_cache_dir,
            verbose=verbose,
        )

    if poisson:
        tile_w_p = Image.open(panels[0]).size[0]
        # With feather, the seams on disk sit at (tile_w - feather_px)
        # increments, not tile_w. Compute the on-disk seam stride so
        # Poisson targets the right columns.
        seam_stride = tile_w_p - feather_px
        poisson_path = out_path.with_name(f"{out_path.stem}-poisson{out_path.suffix}")
        poisson_blend_seams(
            tapestry_path=out_path,
            out_path=poisson_path,
            cols=cols,
            rows=rows,
            tile_w=seam_stride,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "family",
        type=Path,
        nargs="+",
        help="one or more family JSON files (each provides title, seed, "
        "grid, and paragraphs). Style comes from --style.",
    )
    parser.add_argument(
        "--style",
        type=Path,
        required=True,
        help="path to a style JSON (e.g. styles/rembrandt.json). Its "
        "style_suffix is appended to every generated prompt; its name (or "
        "filename stem) is used in the cache path and default output name.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output image path (only valid with a single family file; "
        "otherwise each family writes <family-stem>-<style-stem>.jpg in "
        "--out-dir)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="directory for per-family outputs when multiple family files "
        "are given (default: current directory)",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("cache"),
        help="parent directory for caches. Layout: "
        "<cache-root>/<family>/prompts/<id>.txt (shared across styles) and "
        "<cache-root>/<family>/images/<style>/<id>.jpg (default: ./cache)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="render only the first N stories from each file and fit them "
        "into a near-square grid. The JSON's base seed is still used, so "
        "the limited output is a byte-exact prefix of an --all run; later "
        "rerunning with --all reuses these cached panels (default: 4).",
    )
    parser.add_argument(
        "--all",
        dest="all_stories",
        action="store_true",
        help="render all stories and use the grid declared in the JSON. "
        "Overrides --limit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print the Gemini-generated prompt for each panel, the full "
        "Nunchaku request parameters, and a per-panel timing table at the "
        "end of each tapestry.",
    )
    parser.add_argument(
        "--no-labels",
        dest="labels",
        action="store_false",
        help="skip the numbered caption bar on each panel. Labels are a "
        "post-processing overlay (not baked into the cached images), so "
        "this only affects the final tapestry.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=None,
        help="path to a TrueType font for panel labels. Defaults to the "
        "first match among common system paths (Arial on macOS, DejaVu "
        "Sans on Linux), falling back to PIL's bitmap default.",
    )
    parser.add_argument(
        "--blend",
        action="store_true",
        help="after assembling the tapestry, run a second pass through the "
        "configured image-edit model to smooth the vertical seams where "
        "panels meet. Writes <stem>-blended.jpg alongside the raw tapestry. "
        "Costs one edit call per vertical seam segment (9 for a 4x3 grid, "
        "2 for a 2x2).",
    )
    parser.add_argument(
        "--edit-provider",
        dest="edit_provider",
        choices=("nunchaku", "gemini"),
        default="nunchaku",
        help="backend used for --blend seam edits. "
        "'nunchaku' uses nunchaku-qwen-image-edit (fast, cheap, often "
        "conservative). 'gemini' uses gemini-3.1-flash-image-preview "
        "(better at 'preserve content, only fix seam' instructions; "
        "somewhat slower). Default: nunchaku.",
    )
    parser.add_argument(
        "--poisson",
        action="store_true",
        help="deterministic alternative to --blend: after assembling the "
        "tapestry, run OpenCV gradient-domain (Poisson) seamless cloning "
        "on a strip straddling every vertical seam. No LLM, no API cost, "
        "milliseconds per seam. Writes <stem>-poisson.jpg. Requires "
        "opencv-python (imported lazily).",
    )
    parser.add_argument(
        "--feather",
        dest="feather_px",
        type=int,
        default=0,
        metavar="PX",
        help="overlap adjacent columns by PX pixels during assembly and "
        "cross-dissolve them with a horizontal alpha gradient. 0 disables "
        "(default). Try 64-128 to soften seams. Narrows the output canvas "
        "by (cols-1)*PX pixels. Unlike --blend and --poisson, this "
        "changes the main <stem>.jpg, not a sibling file.",
    )
    parser.add_argument(
        "--frame",
        dest="frame_px",
        type=int,
        default=12,
        metavar="PX",
        help="paint a two-tone (dark outer + cream inner) frame PX "
        "pixels thick around every tile before pasting. Default: 12 "
        "(understated panel borders). Pass 0 to disable, or a larger "
        "value (e.g. 20, 32) for a heavier frame. Turns visible seams "
        "into deliberate panel dividers.",
    )
    parser.add_argument(
        "--full-blend",
        dest="full_blend",
        action="store_true",
        help="shortcut for the recommended deterministic stack: enables "
        "--poisson and sets --feather to 96 unless you passed a "
        "different --feather explicitly. Keeps whatever --frame you "
        "chose (default 12). Costs nothing extra vs running the flags "
        "separately.",
    )
    parser.add_argument(
        "--regenerate-images",
        dest="regenerate_images",
        action="store_true",
        help="before rendering, wipe the cached images (and seam "
        "patches) for the selected (family, style) so every panel is "
        "re-rendered from scratch. Prompts are preserved, so Gemini is "
        "not re-called — only Nunchaku. Useful after you change the "
        "story paragraphs or the 'people' list and want fresh images "
        "without manually deleting cache directories.",
    )
    args = parser.parse_args()
    if args.full_blend:
        args.poisson = True
        if args.feather_px == 0:
            args.feather_px = 96
    limit = None if args.all_stories else args.limit

    if args.out is not None and len(args.family) > 1:
        print(
            "error: --out only valid with a single family file; "
            "use --out-dir for multiple families",
            file=sys.stderr,
        )
        return 2

    if not args.style.exists():
        print(f"error: style file {args.style} not found", file=sys.stderr)
        return 2

    for p in args.family:
        if not p.exists():
            print(f"error: family file {p} not found", file=sys.stderr)
            return 2

    style_name, style_suffix = load_style(args.style)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    nunchaku_key = os.environ["NUNCHAKU_API_KEY"]

    overall_start = time.monotonic()
    for family_path in args.family:
        if args.out is not None:
            out_path = args.out
        else:
            out_path = args.out_dir / f"{family_path.stem}-{style_name}.jpg"
        build_one(
            stories_path=family_path,
            out_path=out_path,
            cache_root=args.cache_root,
            style_name=style_name,
            style_suffix=style_suffix,
            gemini_client=gemini_client,
            nunchaku_key=nunchaku_key,
            limit=limit,
            verbose=args.verbose,
            labels=args.labels,
            font_path=args.font,
            blend=args.blend,
            edit_provider=args.edit_provider,
            feather_px=args.feather_px,
            frame_px=args.frame_px,
            poisson=args.poisson,
            regenerate_images=args.regenerate_images,
        )
    if len(args.family) > 1:
        print(
            f"all {len(args.family)} tapestries done "
            f"(total {time.monotonic() - overall_start:.0f}s)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
