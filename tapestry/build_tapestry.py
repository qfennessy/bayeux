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
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import requests
from google import genai
from PIL import Image, ImageDraw, ImageFont


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
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

PROMPT_SYSTEM_INSTRUCTION = (
    "You convert a paragraph of family history into a single concise visual "
    "image prompt for a text-to-image model. Describe ONE scene with concrete "
    "visual elements: who is present, what they are doing, the setting, key "
    "objects, time of day, and mood. Do not name specific real people. Do not "
    "include camera or lens jargon. Output only the prompt as one paragraph of "
    "roughly 50-80 words. Do not add preamble, quotes, or labels."
)


def gemini_prompt_for(client: genai.Client, paragraph: str) -> str:
    """Ask Gemini 3.1 Flash to turn a paragraph into a visual prompt.

    Thinking is disabled (``thinking_budget=0``): paragraph-to-prompt
    rewriting doesn't need reasoning, and leaving it on returns a
    ``thought_signature`` part that the SDK warns about and that costs
    extra latency and tokens.
    """
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=paragraph,
        config={
            "system_instruction": PROMPT_SYSTEM_INSTRUCTION,
            "thinking_config": {"thinking_budget": 0},
        },
    )
    return resp.text.strip()


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

    Prompts are cached under ``prompt_dir`` (style-independent — Gemini
    only sees the family paragraph) and images under ``image_dir`` (per
    style). This lets multiple styles of the same family share the
    Gemini work and only re-render the Nunchaku image.

    Returns (image_path, stats). Stats keys: ``prompt_seconds``,
    ``image_seconds`` (0.0 when served from cache), ``prompt_cached``,
    ``image_cached``, ``prompt_chars``, ``image_bytes``, ``seed``.
    """
    sid = story["id"]
    prompt_path = prompt_dir / f"{sid}.txt"
    image_path = image_dir / f"{sid}.jpg"
    stats: dict = {
        "prompt_seconds": 0.0,
        "image_seconds": 0.0,
        "prompt_cached": False,
        "image_cached": False,
        "prompt_chars": 0,
        "image_bytes": 0,
        "seed": base_seed + index,
    }

    if prompt_path.exists():
        visual_prompt = prompt_path.read_text().strip()
        stats["prompt_cached"] = True
        stats["prompt_chars"] = len(visual_prompt)
        print(f"  prompt: cached ({len(visual_prompt)} chars)", flush=True)
    else:
        t0 = time.monotonic()
        print(f"  prompt: gemini...", flush=True)
        visual_prompt = gemini_prompt_for(gemini_client, story["paragraph"])
        elapsed = time.monotonic() - t0
        prompt_path.write_text(visual_prompt + "\n")
        stats["prompt_seconds"] = elapsed
        stats["prompt_chars"] = len(visual_prompt)
        print(
            f"  prompt: ok ({elapsed:.2f}s, {len(visual_prompt)} chars, "
            f"model {GEMINI_MODEL})",
            flush=True,
        )
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
    """Return a copy of ``tile`` with ``text`` drawn in a translucent bar.

    The bar is ~8% of the image height, pinned to the bottom, with
    centered white text over a 55%-opaque black fill. Operates on a
    copy so callers can reuse the original (cached) image object.
    """
    w, h = tile.size
    bar_h = max(1, int(h * 0.08))
    overlay = Image.new("RGBA", tile.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([(0, h - bar_h), (w, h)], fill=(0, 0, 0, 140))
    text_w = draw.textlength(text, font=font)
    # Place text roughly centered vertically inside the bar.
    text_y = h - bar_h + (bar_h - font.size) // 2 if hasattr(font, "size") else h - bar_h
    draw.text(
        ((w - text_w) / 2, text_y),
        text,
        fill=(255, 255, 255, 255),
        font=font,
    )
    base = tile.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def assemble_tapestry(
    panels: list[Path],
    cols: int,
    rows: int,
    out_path: Path,
    labels: list[str] | None = None,
    font_path: Path | None = None,
) -> None:
    """Paste all panels into a cols x rows grid and save.

    When ``labels`` is provided (one string per panel), each tile gets a
    translucent caption bar at the bottom before being pasted. Labels are
    only drawn on the final tapestry, never on the cached per-panel
    images, so you can change or disable them without regenerating.
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
    canvas = Image.new("RGB", (cols * w, rows * h), "white")
    font = _load_font(int(h * 0.045), font_path) if labels else None
    for i, tile in enumerate(tiles):
        if tile.size != (w, h):
            tile = tile.resize((w, h), Image.LANCZOS)
        if labels:
            tile = label_tile(tile, labels[i], font)
        r, c = divmod(i, cols)
        canvas.paste(tile, (c * w, r * h))
    canvas.save(out_path, quality=92)


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
        # Extension point: a future per-story "label" field wins over the
        # numbered default. For now, stories rarely carry one.
        label_texts = [
            story.get("label") or f"Image {i + 1}"
            for i, story in enumerate(stories)
        ]
    else:
        label_texts = None
    assemble_tapestry(
        panels, cols, rows, out_path, labels=label_texts, font_path=font_path
    )
    print(
        f"wrote {out_path} (total {time.monotonic() - run_start:.0f}s)",
        flush=True,
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
    args = parser.parse_args()
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
