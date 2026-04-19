# bayeux

*Created: 2025-11-15 · Last edited: 2026-04-19*

Build a Bayeux-tapestry-style image grid from a JSON of family-history paragraphs.

Given paragraphs like *"In the autumn of 1880, Tommaso and Lucia Ricci stepped off the steamship Canopic at Castle Garden…"*, Gemini 3.1 Flash Lite rewrites each into a visual prompt, Nunchaku Qwen-Image renders it at a chosen style, and the script assembles every panel into a single labeled, optionally seam-blended tapestry.

## Where the code lives

See [`tapestry/README.md`](tapestry/README.md) for the full schema (family JSONs × style JSONs), CLI flags, cache layout, panel-label and seam-blending options, and worked examples.

```bash
pip install -r requirements.txt

# Put NUNCHAKU_API_KEY and GEMINI_API_KEY in tapestry/.env.
set -a && source tapestry/.env && set +a

python -u tapestry/build_tapestry.py \
    tapestry/ricci-bradford-1880.json \
    --style tapestry/styles/rembrandt.json
```

## Layout

```
tapestry/              the tapestry app (main code)
  build_tapestry.py
  README.md
  *.json               family files: paragraphs + seed + grid
  styles/*.json        style files: one style_suffix per visual look
examples/python/       minimal API references the pipeline is built on
  text_to_image.py
  image_to_image.py
requirements.txt       deps for the tapestry app
```

Other directories (`demo/`, `tests/`, video/JS/cURL starters under `examples/`) are left over from the original Nunchaku starter kit. They are gitignored but kept on disk as a local reference.

## Acknowledgements

This project began as Nunchaku's Python/cURL/JS starter kit and has been narrowed to the tapestry pipeline.
