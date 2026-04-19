<p align="center">
  <img src="assets/Bayeux-hero.jpg" alt="Detail from the Bayeux Tapestry — the project namesake" width="720">
</p>

# bayeux

*Created: 2026-04-19*

Build a Bayeux-tapestry-style image grid from a JSON of family-history paragraphs.

Given paragraphs like *"In the autumn of 1880, Tommaso and Lucia Ricci stepped off the steamship Canopic at Castle Garden…"*, Gemini 3.1 Flash Lite rewrites each into a visual prompt, Nunchaku `nunchaku-qwen-image` renders it at a chosen style, and the script assembles twelve panels into a single labeled, optionally seam-blended tapestry. A small static-site generator walks the rendered output and produces a read-only Vercel gallery.

## Why this exists

Family history can be represented with creative graphics. A small AI pipeline fixes that: Gemini 3.1 Flash Lite converts each paragraph into a concrete visual scene prompt, a shared style suffix ties every panel together visually, and a deterministic `base_seed + index` keeps reruns byte-identical while still varying per panel.

**Family** (who / what happened) and **style** (how it looks) are orthogonal. Families live in `tapestry/*.json`; styles in `tapestry/styles/*.json`. Any family can be rendered in any style without editing either file.

## Quick start

```bash
pip install -r requirements.txt

# Put NUNCHAKU_API_KEY and GEMINI_API_KEY in .env.
set -a && source .env && set +a

# 2x2 preview (default --limit 4) of one family in one style:
python -u tapestry/build_tapestry.py \
    tapestry/ricci-bradford-1880.json \
    --style tapestry/styles/rembrandt.json

# Full 4x3 tapestry with feathered cross-dissolve + Poisson seam smoothing:
python -u tapestry/build_tapestry.py \
    tapestry/ricci-bradford-1880.json \
    --style tapestry/styles/rembrandt.json \
    --all --full-blend --out-dir out
```

## Pipeline

For each paragraph in a family JSON:

1. **Gemini 3.1 Flash Lite** (`gemini-3.1-flash-lite-preview`) reads the paragraph plus a `people` list and returns structured JSON with (a) a short narrative title (year-prefixed when the story has a `year`) and (b) a concise visual prompt.
2. The shared `style_suffix` from the style JSON is appended so every panel shares one cohesive look.
3. **Nunchaku `nunchaku-qwen-image`** at `tier: radically_fast` renders the panel at the configured size (default `768x768`) using seed `base_seed + index`.
4. **Pillow** pastes all panels into a `cols × rows` grid, overlays white outlined labels from the Gemini titles, and writes the output JPG. Optional post-processing (feather, frame, Poisson, edit-model blend) runs from there.

## Family JSON schema

```json
{
  "title": "The Ricci-Bradford Chronicle",
  "seed": 18801906,
  "grid": {"cols": 4, "rows": 3},
  "image_size": "768x768",
  "stories": [
    {
      "id": "01-castle-garden-arrival",
      "category": "immigration",
      "year": "1880",
      "people": ["Tommaso Ricci", "Lucia Ricci", "Salvatore Ricci (infant)"],
      "paragraph": "..."
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `title` | Printed at run start; also used as the gallery heading. |
| `seed` | Base seed for the set. Per-image seed is `seed + index`, reproducible across reruns. |
| `grid.cols` / `grid.rows` | Full-grid shape when `--all` is used. `len(stories)` must equal `cols * rows`. |
| `image_size` | Passed to Nunchaku. Default `768x768` for web-screen display. |
| `stories[].id` | Filesystem-safe, unique — used as cache filename stem. |
| `stories[].category` | Free-form tag. Not sent to the model. |
| `stories[].year` | Optional. When present, prefixes the generated title (e.g. `"1880 Arrival at Castle Garden"`). |
| `stories[].people` | Optional list of character names / roles. When non-empty, Gemini names each person with a short visual descriptor in the prompt so characters render consistently across panels. |
| `stories[].paragraph` | Prose. Gemini turns it into an image prompt. |

Family JSONs **must not** contain a `style_suffix` — the loader rejects it. Style lives on its own JSON.

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

Shipped styles: `bayeux`, `rembrandt`, `hanna-barbera`, `watercolor`, `ukiyo-e`.

## CLI flags

```bash
python tapestry/build_tapestry.py FAMILY [FAMILY ...] --style STYLE [options]
```

**Required**

- `--style PATH` — a style JSON under `tapestry/styles/`.

**Scope**

- `--limit N` — render only the first N stories into a near-square grid. Default: **4** (a 2×2 sampler). Byte-exact prefix of an `--all` run because the seed is stable.
- `--all` — render every story using the grid declared in the family JSON (e.g. 4×3 for 12 stories).
- `--regenerate-images` — before rendering, wipe the cached panels for this (family, style) so every panel re-renders. Prompts are preserved (Gemini not re-called).

**Output**

- `--out PATH` — output image path. Only valid with a **single** family file. Defaults to `<family-stem>-<style-stem>.jpg`.
- `--out-dir DIR` — directory for per-family outputs. Default: current directory.
- `--cache-root DIR` — parent directory for caches. Default: `./cache/`.

**Assembly** (modify the main `<stem>.jpg`)

- `--frame PX` — two-tone dark-outer + cream-inner border PX pixels thick on every tile. Default: **12**. Pass 0 to disable.
- `--feather PX` — overlap adjacent columns by PX pixels and cross-dissolve with an alpha gradient. 0 disables (default). Narrows the output canvas by `(cols-1)*PX`.
- `--no-labels` — omit the outlined caption at the bottom of each panel.
- `--font PATH` — specific TrueType font for the caption. Defaults to Arial on macOS / DejaVu Sans on Linux.

**Post-process** (write sibling files, main stays untouched)

- `--poisson` — OpenCV gradient-domain (Poisson) seamless cloning on a ±192 px strip at every vertical seam. Writes `<stem>-poisson.jpg`. No API calls; lazy-imports `opencv-python-headless`.
- `--blend` — edit-model smoothing of vertical seams. Writes `<stem>-blended.jpg`. 9 calls per 4×3, 2 per 2×2.
- `--edit-provider nunchaku|gemini` — backend for `--blend`. `nunchaku` is cheap but often too conservative. `gemini` (`gemini-3.1-flash-image-preview`) follows "preserve content, fix only the seam" much better. Default: `nunchaku`.
- `--full-blend` — shortcut: enables `--poisson` and sets `--feather 96` unless you passed a different `--feather`. Leaves `--frame` at whatever you chose. Recommended deterministic default.

**Other**

- `-v`, `--verbose` — prints the Gemini-generated prompt for each panel, request parameters, and a per-panel timing table at run end.

## Cache layout

```
cache/
  <family-stem>/
    prompts/<story-id>.json              # {title, prompt, people, year, instruction_hash}
    images/<style-stem>/<story-id>.jpg   # per (family, style)
    images/<style-stem>/seams/<provider>/<prompt_hash>/x<NNNNN>-y<NNNNN>.jpg
```

Because prompts are style-independent (Gemini only sees the family paragraph), switching styles for the same family **does not re-call Gemini** — only Nunchaku regenerates per style. Seam patches cache by prompt hash, so tweaking the seam prompt auto-invalidates them without disturbing the panels.

Prompt invalidation triggers: a story's `people` list changed, its `year` changed, or the system instruction hash changed. The invalidation reason is logged (`"gemini: cache invalidated (year none→1882, system-instruction f708114e86→dc77e7e77d)"`).

## Label rendering

Each tile gets an outlined white caption (no dark band). Label-text priority:

1. Explicit `story.label` in the family JSON (editor wins).
2. Gemini-generated narrative title — year-prefixed when the story has a `year`, often naming a character from `people`.
3. Fallback `Image N`.

Labels are drawn by `assemble_tapestry` as a post-processing overlay — **they are not baked into the cached JPGs**, so changing the font or toggling `--no-labels` does not invalidate cached images.

## Seam handling

Visible seams between independently-rendered panels are handled at up to three levels:

- **`--feather 96`** during assembly — alpha-blends adjacent tile edges. Deterministic, zero API calls, mild visual crossfade.
- **`--poisson`** post-process — `cv2.seamlessClone(MIXED_CLONE)` on a ±192 px strip at each vertical seam. Deterministic, hides abrupt sky/ground colour jumps. Writes `<stem>-poisson.jpg`.
- **`--blend`** post-process — edit-model pass (Nunchaku or Gemini). For Gemini, a magenta guideline is painted on the seam column before the crop is sent, and the prompt asks the model to erase the line and blend the scene behind it. Nunchaku gets a clean crop with a traditional seam-smoothing prompt (it can't reliably remove the magenta). Retries transient 5xx/429 errors with exponential backoff; each patch is cached to disk so partial failures resume.

`--full-blend` combines `--feather 96` + `--poisson` — deterministic, no API calls beyond the panel generations, and usually the best-looking result.

**`--frame`** is orthogonal and often the cleanest fix: draw a decorative panel border so seams become deliberate dividers instead of defects. Default on (12 px).

## Publishing the gallery

Read-only public gallery deploys via **Vercel** as a static site.

```
public/         static site emitted by tapestry/build_gallery.py
vercel.json     CDN cache headers + cleanUrls
```

**Build the gallery:**

```bash
# Render whichever tapestries you want to publish:
python -u tapestry/build_tapestry.py tapestry/ricci-bradford-1880.json \
    --style tapestry/styles/rembrandt.json --full-blend --out-dir out

# Walk cache + rendered images → public/index.html + public/tapestries/*.jpg:
python tapestry/build_gallery.py --images-dir out --clean
```

`build_gallery.py` picks the most-processed variant per (family, style) — `-poisson` > `-blended` > raw — copies each to `public/tapestries/<family>-<style>.jpg` (clean URL, variant hidden), and embeds per-panel metadata (title, year, people, paragraph, generated prompt) as expandable `<details>` sections. Pure Python + Pillow; no JS toolchain.

`--clean` wipes `public/tapestries/` before repopulating so the deployed dir is an exact mirror of `--images-dir`. Use after removing images from `out/` to prevent orphans from lingering.

**Preview:**

```bash
python -m http.server 8000 -d public
```

**Deploy:**

```bash
npm i -g vercel   # one-time
vercel            # first time: link to a Vercel project
vercel --prod     # deploy
```

No serverless functions, no env vars on Vercel — it's just `public/`. API keys stay local. Re-run, git-commit `public/`, then `vercel --prod` (or push and let Vercel auto-deploy).

## Rate limits and retries

Nunchaku returns **429 Too Many Requests** when the account's rate window is exhausted. The generation and edit paths both retry up to 8 times, honoring `Retry-After` when present and backing off ~20 s per attempt otherwise. If all 8 retries fail the script exits non-zero and leaves finished work in the cache — just rerun once the quota resets. The per-panel image cache and per-patch seam cache make reruns resumable.

Gemini image-edit calls (used by `--edit-provider gemini`) retry 5xx (e.g. *"Deadline expired"*) and 429 with the same policy. Non-retriable 4xx (400/403/404) propagate immediately so auth or model-name mistakes fail loudly.

For a full 30-combo matrix run (6 families × 5 styles × 12 panels), use `tapestry/render_all.sh` — it iterates, skips anything already in `out/`, and is safe to re-invoke until every combo is assembled.

## Benchmarking Nunchaku vs Gemini

`tapestry/benchmark.py` runs the same prompt set through both providers and writes:

```
bench/results.csv                 per-call provider, seconds, bytes, status
bench/images/<id>-nunchaku.jpg
bench/images/<id>-gemini.jpg
bench/index.html                  side-by-side gallery for human quality eval
```

Prompts come from cached Gemini outputs so both providers receive the same concrete visual prompt the pipeline would normally hand to Nunchaku.

```bash
set -a && source .env && set +a
python tapestry/benchmark.py \
    --family tapestry/ricci-bradford-1880.json \
    --family tapestry/chen-vasquez-1865.json \
    --count 12 --size 768x768
```

Flags: `--count N` (default 12), `--size WxH` (default 768x768), `--seed N` (base seed for Nunchaku; Gemini has no exposed seed), `--skip-if-cached` to extend a prior run without re-billing.

Open `bench/index.html` for the pairwise viewer — each prompt card shows the source prompt, Nunchaku on the left, Gemini on the right, with per-call latency and file size captions. The top of the page carries a summary table of median/p95 latency and average byte size per provider.

## Notes on model choices

- `gemini-3.1-flash-lite-preview` — the 3.1 Flash family ships only as `-lite-preview`, `-image-preview`, `-tts-preview`, `-live-preview`; there is no bare `gemini-3.1-flash`. Lite is the right choice for cheap, fast text rewriting.
- `gemini-3.1-flash-image-preview` — used by `--edit-provider gemini`. Significantly better instruction-following on "preserve content, only fix the seam" than diffusion-based edit models.
- `nunchaku-qwen-image` at `tier: radically_fast` — the fastest tier. `num_inference_steps: 28` (image) / `40` (edit) are the defaults we landed on.
- `nunchaku-qwen-image-edit` — conservative by design, so seam blending via `--edit-provider nunchaku` often produces near-identity transforms. `--poisson` is more reliable as a deterministic fallback.

## Layout

```
tapestry/                 the tapestry app (main code)
  build_tapestry.py       pipeline: family JSON + style → tapestry JPG
  build_gallery.py        static site generator → public/
  render_all.sh           render every (family, style) combo
  <family>.json           family files: paragraphs + seed + grid + people
  styles/<style>.json     style files: style_suffix
examples/python/          minimal API references the pipeline is built on
  text_to_image.py
  image_to_image.py
public/                   deployed static site (tracked in git)
assets/Bayeux-hero.jpg    hero banner used by build_gallery.py
docs/images/              curated README / site imagery (tracked)
cache/                    local prompt + panel + seam caches (gitignored)
out/                      rendered tapestry JPGs (gitignored)
requirements.txt          tapestry app deps
vercel.json               CDN cache headers + cleanUrls for the static gallery
```

Other directories (`demo/`, `tests/`, video/JS/cURL starters under `examples/`) are left over from the original Nunchaku starter kit. They are gitignored but kept on disk as a local reference.

## Acknowledgements

Built by Quentin Fennessy at the [Sundai Club](https://sundai.club) on 19 April 2026, as a demonstration of the highly optimized diffusion models from [Nunchaku](https://nunchaku.dev). This project began as Nunchaku's Python/cURL/JS starter kit and has been narrowed to the tapestry pipeline. All family chronicles shown in the gallery are fictional, created to demonstrate the pipeline.

Web URL: https://bayeux.vercel.app
Sundai Project: https://www.sundai.club/projects/b07760aa-fa4c-4a21-9302-9f2050dd846f
Repository: https://github.com/qfennessy/bayeux