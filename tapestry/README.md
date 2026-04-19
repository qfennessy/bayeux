# Tapestry

*Created: 2026-04-19 · Last edited: 2026-04-19*

> **Family** (who / what happened) and **style** (how it looks) are orthogonal. Families live in `tapestry/*.json`; styles live in `tapestry/styles/*.json`. Pick one of each at the command line — any family can be rendered in any style without editing either file.

Generate a Bayeux-tapestry-style image grid from a JSON of family-history paragraphs.

For each paragraph:

1. **Gemini 3.1 Flash Lite** (`gemini-3.1-flash-lite-preview`) rewrites the paragraph as a single concrete visual prompt.
2. A shared `style_suffix` is appended so every panel shares one cohesive look.
3. **Nunchaku `nunchaku-qwen-image`** (`tier: radically_fast`) renders the panel at 1024×1024 using a deterministic seed.
4. Pillow pastes all panels into a `cols × rows` grid and writes one output image.

## Why this exists

The input format is paragraphs of family history, which read well but are bad image prompts (too long, abstract, full of names and dates). Gemini 3.1 Flash Lite converts each one into a concrete visual scene prompt cheaply and quickly. The shared `style_suffix` is what makes the grid look like a single tapestry instead of twelve unrelated pictures, and the `base_seed + index` scheme makes each reproduce byte-for-byte on rerun while still varying between panels.

## Files

- `<family>.json` — one file per family / story set (e.g. `ricci-bradford-1880.json`). Contains title, seed, grid, image size, and the ordered paragraphs. **Contains no style.**
- `styles/<style>.json` — one file per visual style (e.g. `styles/rembrandt.json`). Contains `name`, `description`, and `style_suffix`. Shipped: `bayeux`, `rembrandt`, `hanna-barbera`, `watercolor`.
- `build_tapestry.py` — the pipeline. Takes one or more families and exactly one style per invocation.
- `requirements.txt` — `requests`, `Pillow`, `google-genai`.
- `.env` — local API keys (`NUNCHAKU_API_KEY`, `GEMINI_API_KEY`). Not committed.

## Family JSON schema

```json
{
  "title": "The Ricci-Bradford Chronicle",
  "seed": 18801906,
  "grid": {"cols": 4, "rows": 3},
  "image_size": "1024x1024",
  "stories": [
    {"id": "01-castle-garden-arrival", "category": "immigration", "paragraph": "..."},
    "... 11 more ..."
  ]
}
```

| Field | Meaning |
|---|---|
| `title` | Printed at the start of the run; not baked into the image. |
| `seed` | Base seed for the whole set. Per-image seed is `seed + index`, reproducible across reruns. |
| `grid.cols` / `grid.rows` | Full-grid shape when `--all` is used. `len(stories)` must equal `cols * rows`. |
| `image_size` | Passed to Nunchaku. Default `1024x1024`. |
| `stories[].id` | Filesystem-safe, unique — used as cache filename stem. |
| `stories[].category` | Free-form tag. Not sent to the model. |
| `stories[].paragraph` | Prose. Gemini turns it into an image prompt. |

Family JSONs **must not** contain a `style_suffix` — the loader rejects it. That field now lives on the style.

## Style JSON schema

```json
{
  "name": "rembrandt",
  "description": "Dutch Golden Age oil painting with chiaroscuro lighting.",
  "style_suffix": "in the style of a Rembrandt oil painting, dark chiaroscuro lighting ..."
}
```

| Field | Meaning |
|---|---|
| `name` | Short identifier used in cache paths and output filenames. Falls back to the file stem if omitted. |
| `description` | Human-readable note. Not sent to the model. |
| `style_suffix` | Appended to every Gemini-generated prompt before hitting Nunchaku. Required. |

## Running

```bash
cd tapestry
pip install -r requirements.txt

# Put keys in tapestry/.env (one per line, KEY=value) or export directly.
set -a && source .env && set +a

# Single family × one style (2x2 preview by default):
python -u build_tapestry.py \
    ricci-bradford-1880.json \
    --style styles/rembrandt.json

# Same family, same paragraphs, different styles — Gemini is called once,
# Nunchaku runs for each style:
python -u build_tapestry.py ricci-bradford-1880.json --style styles/bayeux.json
python -u build_tapestry.py ricci-bradford-1880.json --style styles/hanna-barbera.json
python -u build_tapestry.py ricci-bradford-1880.json --style styles/watercolor.json

# Many families in a single invocation, all in the same style:
python -u build_tapestry.py \
    ricci-bradford-1880.json \
    chen-vasquez-1865.json \
    nakamura-kealoha-1908.json \
    --style styles/rembrandt.json \
    --out-dir out/
```

Flags:

- `--style PATH` — **required.** Path to a style JSON under `styles/`.
- `--limit N` — render only the first N stories and auto-fit them into a near-square grid. **Default: 4** (a 2×2 sampler). Limited output is a byte-exact prefix of an `--all` run because the seed is stable.
- `--all` — ignore `--limit` and render every story using the grid declared in the family JSON (e.g. 4×3 for 12 stories).
- `-v`, `--verbose` — prints the Gemini-generated prompt for each panel, the Nunchaku request parameters, and a per-panel timing table at the end of each tapestry.
- `--no-labels` — omit the caption bar at the bottom of each panel.
- `--font PATH` — use a specific TrueType font for the labels. Defaults to Arial on macOS / DejaVu Sans on Linux.
- `--blend` — after assembling, run a second pass through the configured image-edit model to smooth **vertical** seams where panels meet. Horizontal seams are skipped by default because the caption bar sits on them. Writes `<stem>-blended.jpg` alongside the raw tapestry. Costs one edit call per vertical seam segment (9 for a 4×3 grid, 2 for a 2×2).
- `--edit-provider nunchaku|gemini` — which backend handles the `--blend` edits. `nunchaku` uses `nunchaku-qwen-image-edit` (fast, cheap, tends to be too conservative to fully erase seams). `gemini` uses `gemini-3.1-flash-image-preview` (much better at "preserve content, only fix the seam"; slightly slower per call). Default: `nunchaku`.

### Panel labels

Each tile gets a translucent caption bar (~8% of tile height) across the bottom. Label text priority:

1. Explicit `"label"` in the story (family JSON wins — lets you override the LLM).
2. Gemini-generated narrative title (3-6 words, Title Case, e.g. *Arrival at Castle Garden*) — produced on the same API call as the visual prompt via structured JSON output, so there's no second round-trip.
3. Fallback `Image N` if no title was ever cached.

Labels are drawn by `assemble_tapestry` as a post-processing overlay — **they are not baked into the cached JPGs**, so toggling `--no-labels`, changing the font, or tweaking the overlay style later does not invalidate cached images.

### Seam blending (`--blend`)

The assembled tapestry shows visible seams where independently-rendered panels meet. `--blend` runs a second pass that:

1. Walks every **vertical** seam in the grid (horizontal seams are skipped so the caption bar at the bottom of each tile stays pristine — the edit model otherwise smears label glyphs).
2. Crops a 1024×1024 patch straddling each seam segment.
3. Sends each patch through `nunchaku-qwen-image-edit` (`radically_fast` tier) with a seam-smoothing prompt that explicitly tells the model to leave horizontal bands and captions alone.
4. Feather-pastes the edited patch back into the tapestry (256px Gaussian-blurred rectangular mask) so patch boundaries dissolve.

For a 4×3 grid this is 3 vertical seams × 3 row segments = **9 edit calls** per tapestry. For a 2×2 preview (`--limit 4` default) it's 1 × 2 = **2 calls**. To re-enable horizontal seam blending (e.g. if you're rendering without labels), flip `vertical_only=False` in `blend_seams`.
- `--out PATH` — output image path. Only valid with a **single** family file. Defaults to `<family-stem>-<style-stem>.jpg`.
- `--out-dir DIR` — directory for per-family outputs when multiple families are given. Each writes `<family-stem>-<style-stem>.jpg`. Default: current directory.
- `--cache-root DIR` — parent directory for caches. Default: `./cache/`.

### Cache layout

```
cache/
  <family-stem>/
    prompts/<story-id>.txt             # shared across styles
    images/<style-stem>/<story-id>.jpg # per (family, style)
```

Because prompts are style-independent (Gemini only sees the family paragraph), switching styles for the same family **does not re-call Gemini**. Only the Nunchaku images are regenerated per style.

### Quota-friendly default

The `--limit 4` default keeps a typical three-family run at 12 images (comfortably under most Nunchaku per-hour quotas) and builds a 2×2 preview tapestry per family. Once you like a style, rerun the same file with `--all` to fill in the remaining panels — only the new ones hit the API, and the 4 you already rendered are reused verbatim because the seed is stable.

## Caching and reruns

- **API failures are safe** — finished panels are never re-billed.
- **Re-prompting one panel**: delete `cache/<family>/prompts/<id>.txt` *and* every `cache/<family>/images/*/<id>.jpg` (a stale prompt paired with a fresh Gemini call would desync them).
- **Re-rendering just one style**: `rm -rf cache/<family>/images/<style>/`. Prompts are kept, so only Nunchaku gets called.
- **Fresh run for one family**: `rm -rf cache/<family>/`. Other families untouched.

## Progress output

The script prints per-panel progress to stdout (unbuffered when run with `python -u`):

```
=== ricci-bradford-1880.json × rembrandt ===
tapestry: The Ricci-Bradford Chronicle | 12 panels | grid 4x3 | prompts cache/ricci-bradford-1880/prompts | images cache/ricci-bradford-1880/images/rembrandt | out ricci-bradford-1880-rembrandt.jpg
[1/12] 01-castle-garden-arrival (t+0s)
  prompt: gemini...
  prompt: ok (0.8s, 412 chars)
  image:  nunchaku seed=18801906...
  image:  ok (11.3s, 389 KB)
  panel done in 12.1s
```

Retry lines appear under `image:` when Nunchaku returns 429 or 5xx.

## Rate limits

Nunchaku returns **429 Too Many Requests** when the account's rate window is exhausted. The script retries up to 8 times, honoring `Retry-After` when the server sends it, and backing off ~20s per attempt otherwise. If all 8 retries fail the script exits non-zero and leaves everything it did finish in the cache — just rerun once the window has reset.

If you hit this consistently, check your Nunchaku plan's per-minute / per-hour image quota; the `radically_fast` tier is faster per call but does not increase the quota.

## Notes on the model choices

- `gemini-3.1-flash-lite-preview` — the only bare `gemini-3.1-flash` name does not exist in the public Gemini API; the 3.1 Flash family ships only as `-lite-preview`, `-image-preview`, `-tts-preview`, and `-live-preview` variants. `-lite-preview` is the right choice for cheap, fast text generation.
- `nunchaku-qwen-image` at `tier: radically_fast` — fastest tier available for this model. `num_inference_steps: 28` is kept at the endpoint's sample-code default.
