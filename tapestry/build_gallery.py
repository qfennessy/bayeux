"""Build a static gallery of every rendered tapestry.

Walks ``tapestry/*.json`` for family files and ``tapestry/styles/*.json``
for styles, finds matching rendered ``<family>-<style>.jpg`` files in
``--images-dir`` (preferring ``-poisson.jpg`` over ``-blended.jpg`` over
the raw tapestry), loads per-panel metadata from
``cache/<family>/prompts/*.json``, and writes a static site to
``--output-dir`` ready to push to Vercel:

    public/
      index.html
      tapestries/
        <family>-<style>.jpg
        ...

No JS build step. Pure Python + Pillow. Re-runnable and deterministic —
every rebuild produces the same HTML given the same inputs, which means
the generated ``public/`` directory is diffable in git and deploys
reproducibly.

Usage:
    python tapestry/build_gallery.py
    python tapestry/build_gallery.py --images-dir out --output-dir public
"""

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TAPESTRY_DIR = REPO_ROOT / "tapestry"
STYLES_DIR = TAPESTRY_DIR / "styles"
CACHE_DIR = REPO_ROOT / "cache"

VARIANT_PRIORITY = ("-poisson", "-blended", "")


def find_output_jpg(
    images_dir: Path, family_stem: str, style_stem: str
) -> tuple[Path | None, str]:
    """Return (preferred_jpg, variant_label) for a (family, style) pair.

    Priority: ``-poisson`` > ``-blended`` > raw. ``variant_label`` is
    one of ``"poisson"``, ``"blended"``, ``"raw"`` for display in the
    gallery.
    """
    for suffix in VARIANT_PRIORITY:
        candidate = images_dir / f"{family_stem}-{style_stem}{suffix}.jpg"
        if candidate.exists():
            return candidate, suffix.lstrip("-") or "raw"
    return None, ""


def load_panels(family_stem: str, stories: list[dict]) -> list[dict]:
    """Merge each story from the family JSON with its cached Gemini output."""
    prompts_dir = CACHE_DIR / family_stem / "prompts"
    panels: list[dict] = []
    for story in stories:
        sid = story["id"]
        entry = {
            "id": sid,
            "paragraph": story.get("paragraph", ""),
            "year": story.get("year", ""),
            "people": story.get("people", []),
            "category": story.get("category", ""),
            "title": "",
            "prompt": "",
        }
        cache_path = prompts_dir / f"{sid}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            entry["title"] = cached.get("title", "")
            entry["prompt"] = cached.get("prompt", "")
        panels.append(entry)
    return panels


def e(s: str) -> str:
    """Shorthand for HTML-escape."""
    return html.escape(s, quote=True)


CSS = """
:root {
  --bg: #faf6ee;
  --ink: #1a1a1a;
  --muted: #6b6257;
  --accent: #7a2e20;
  --rule: #d9cfb9;
  --card: #ffffff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: ui-serif, Georgia, "Iowan Old Style", "Times New Roman", serif;
  font-size: 16px;
  line-height: 1.55;
}
header {
  padding: 3rem 1.5rem 1.5rem;
  text-align: center;
  border-bottom: 1px solid var(--rule);
}
header h1 {
  font-family: ui-serif, Georgia, serif;
  font-size: 3rem;
  margin: 0 0 0.5rem;
  letter-spacing: 0.02em;
}
header p { margin: 0; color: var(--muted); }
header .links {
  margin-top: 1rem;
  font-size: 0.95rem;
  display: inline-flex;
  gap: 1.25rem;
  align-items: baseline;
}
header .links a {
  color: var(--accent);
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: border-color 0.12s ease;
}
header .links a:hover { border-bottom-color: var(--accent); }
.sponsors {
  max-width: 720px;
  margin: 1.75rem auto 0;
  padding: 1.25rem 1.5rem;
  background: var(--accent);
  color: #fff;
  border-radius: 4px;
  text-align: center;
}
.sponsors .label {
  font-size: 0.7rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  opacity: 0.8;
  margin: 0 0 0.5rem;
}
.sponsors .logos {
  font-size: 1.4rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  margin: 0 0 0.75rem;
}
.sponsors .logos a {
  color: #fff;
  text-decoration: none;
  border-bottom: 2px solid rgba(255, 255, 255, 0.5);
  padding-bottom: 2px;
  transition: border-color 0.12s ease;
}
.sponsors .logos a:hover { border-bottom-color: #fff; }
.sponsors .logos .sep {
  display: inline-block;
  margin: 0 0.75rem;
  opacity: 0.6;
  font-weight: 300;
}
.sponsors .pitch {
  margin: 0;
  font-size: 0.95rem;
  font-style: italic;
  line-height: 1.5;
  opacity: 0.95;
}
.sponsors .pitch a {
  color: #fff;
  text-decoration: underline;
  text-decoration-color: rgba(255, 255, 255, 0.55);
  text-underline-offset: 2px;
}
main { max-width: 1200px; margin: 0 auto; padding: 2rem 1rem 4rem; }
.tapestry {
  margin: 0 0 4rem;
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 4px;
  overflow: hidden;
}
.tapestry h2 {
  margin: 0;
  padding: 1rem 1.25rem;
  border-bottom: 1px solid var(--rule);
  font-size: 1.5rem;
  font-weight: 600;
}
.tapestry h2 .style {
  display: inline-block;
  font-size: 0.85rem;
  color: var(--muted);
  font-weight: 400;
  margin-left: 0.5rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.tapestry h2 .variant {
  display: inline-block;
  font-size: 0.7rem;
  color: var(--accent);
  font-weight: 500;
  margin-left: 0.5rem;
  padding: 0.1rem 0.5rem;
  border: 1px solid var(--accent);
  border-radius: 2px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  vertical-align: middle;
}
.tapestry .hero { display: block; width: 100%; height: auto; }
.tapestry details { border-top: 1px solid var(--rule); }
.tapestry > details > summary {
  padding: 0.75rem 1.25rem;
  cursor: pointer;
  font-weight: 600;
  color: var(--accent);
  list-style: none;
}
.tapestry > details > summary::-webkit-details-marker { display: none; }
.tapestry > details > summary::before { content: "▸ "; }
.tapestry > details[open] > summary::before { content: "▾ "; }
.panels {
  list-style: none;
  margin: 0;
  padding: 0;
  border-top: 1px solid var(--rule);
}
.panel {
  padding: 1rem 1.25rem;
  border-bottom: 1px solid var(--rule);
}
.panel:last-child { border-bottom: 0; }
.panel h3 {
  margin: 0 0 0.25rem;
  font-size: 1.1rem;
  font-weight: 600;
}
.panel .year { color: var(--accent); font-weight: 700; }
.panel .meta {
  font-size: 0.85rem;
  color: var(--muted);
  margin: 0 0 0.5rem;
}
.panel .paragraph { margin: 0 0 0.5rem; }
.panel details summary {
  font-size: 0.85rem;
  color: var(--accent);
  cursor: pointer;
}
.panel details summary::-webkit-details-marker { display: none; }
.panel details summary::before { content: "▸ "; }
.panel details[open] summary::before { content: "▾ "; }
.panel details p {
  font-family: ui-monospace, Menlo, Consolas, monospace;
  font-size: 0.85rem;
  background: var(--bg);
  padding: 0.75rem;
  border-radius: 3px;
  margin: 0.5rem 0 0;
  white-space: pre-wrap;
}
footer {
  text-align: center;
  padding: 2rem 1.5rem;
  color: var(--muted);
  font-size: 0.9rem;
  border-top: 1px solid var(--rule);
  line-height: 1.7;
}
footer a { color: var(--accent); text-decoration: none; }
footer a:hover { text-decoration: underline; }
footer .tagline { font-style: italic; margin-top: 0.5rem; }
""".strip()


def render_panel(panel: dict, idx: int) -> str:
    title = panel["title"] or f"Panel {idx + 1}"
    year = panel["year"]
    people = panel["people"]
    paragraph = panel["paragraph"]
    prompt = panel["prompt"]

    people_str = ", ".join(people) if people else "—"
    meta_parts = []
    if year:
        meta_parts.append(f'<span class="year">{e(str(year))}</span>')
    if panel["category"]:
        meta_parts.append(e(panel["category"]))
    meta_parts.append(f"people: {e(people_str)}")
    meta_html = " · ".join(meta_parts)

    prompt_html = (
        f"<details><summary>Image prompt</summary>"
        f"<p>{e(prompt)}</p></details>"
        if prompt
        else ""
    )

    return (
        f'<li class="panel">\n'
        f"  <h3>{e(title)}</h3>\n"
        f'  <p class="meta">{meta_html}</p>\n'
        f'  <p class="paragraph">{e(paragraph)}</p>\n'
        f"  {prompt_html}\n"
        f"</li>"
    )


def render_tapestry(entry: dict) -> str:
    panel_html = "\n".join(
        render_panel(p, i) for i, p in enumerate(entry["panels"])
    )
    variant_badge = (
        f'<span class="variant">{e(entry["variant"])}</span>'
        if entry["variant"] != "raw"
        else ""
    )
    return (
        f'<section class="tapestry">\n'
        f"  <h2>{e(entry['family_title'])}"
        f' <span class="style">{e(entry["style_name"])}</span>'
        f"  {variant_badge}</h2>\n"
        f'  <img class="hero" loading="lazy" '
        f'src="tapestries/{e(entry["jpg_name"])}" '
        f'alt="{e(entry["family_title"])} rendered in {e(entry["style_name"])} style">\n'
        f"  <details>\n"
        f"    <summary>{len(entry['panels'])} panels</summary>\n"
        f'    <ol class="panels">\n{panel_html}\n    </ol>\n'
        f"  </details>\n"
        f"</section>"
    )


def render_index(entries: list[dict]) -> str:
    tapestry_html = "\n".join(render_tapestry(entry) for entry in entries)
    summary = (
        f"{len(entries)} tapestries" if entries else "No tapestries yet"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>bayeux — family-history tapestries</title>
<style>
{CSS}
</style>
</head>
<body>
<header>
  <h1>bayeux</h1>
  <p>Family-history tapestries. {e(summary)}.</p>
  <section class="sponsors" aria-label="Sponsors">
    <p class="label">Sponsored by</p>
    <p class="logos">
      <a href="https://sundai.club">Sundai Club</a>
      <span class="sep" aria-hidden="true">×</span>
      <a href="https://nunchaku.dev">Nunchaku.dev</a>
    </p>
    <p class="pitch">
      A demonstration of the highly optimized diffusion models from
      <a href="https://nunchaku.dev">Nunchaku</a> —
      every panel rendered by <code>nunchaku-qwen-image</code>
      at the <code>radically_fast</code> tier.
    </p>
  </section>
  <p class="links">
    <a href="https://github.com/qfennessy/bayeux">Source on GitHub</a>
    <span aria-hidden="true">·</span>
    <a href="https://sundai.club">sundai.club</a>
    <span aria-hidden="true">·</span>
    <a href="https://nunchaku.dev">nunchaku.dev</a>
  </p>
</header>
<main>
{tapestry_html}
</main>
<footer>
  <div>
    Sponsored by <a href="https://sundai.club">Sundai Club</a> and
    <a href="https://nunchaku.dev">Nunchaku.dev</a> ·
    <a href="https://github.com/qfennessy/bayeux">qfennessy/bayeux</a> on GitHub
  </div>
  <div class="tagline">
    Family-history paragraphs → Gemini 3.1 Flash Lite prompts → Nunchaku Qwen-Image panels → tapestry.
  </div>
</footer>
</body>
</html>
"""


def discover_families() -> list[Path]:
    """Family JSONs are every ``tapestry/*.json`` (styles live one level
    deeper under ``tapestry/styles/``)."""
    return sorted(TAPESTRY_DIR.glob("*.json"))


def discover_styles() -> list[Path]:
    return sorted(STYLES_DIR.glob("*.json"))


def build(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    images_dir: Path = args.images_dir
    tapestries_out = output_dir / "tapestries"
    tapestries_out.mkdir(parents=True, exist_ok=True)

    families = discover_families()
    styles = discover_styles()
    if not families:
        print(f"error: no family JSONs found in {TAPESTRY_DIR}", file=sys.stderr)
        return 1
    if not styles:
        print(f"error: no style JSONs found in {STYLES_DIR}", file=sys.stderr)
        return 1

    entries: list[dict] = []
    skipped: list[str] = []
    for family_path in families:
        family_data = json.loads(family_path.read_text())
        family_title = family_data.get("title", family_path.stem)
        panels = load_panels(family_path.stem, family_data.get("stories", []))
        for style_path in styles:
            style_data = json.loads(style_path.read_text())
            style_name = style_data.get("name", style_path.stem)
            jpg, variant = find_output_jpg(
                images_dir, family_path.stem, style_path.stem
            )
            if jpg is None:
                skipped.append(f"{family_path.stem} × {style_name}")
                continue
            dest = tapestries_out / jpg.name
            if (
                not dest.exists()
                or dest.stat().st_mtime < jpg.stat().st_mtime
            ):
                shutil.copy2(jpg, dest)
            entries.append(
                {
                    "family_stem": family_path.stem,
                    "family_title": family_title,
                    "style_name": style_name,
                    "style_description": style_data.get("description", ""),
                    "jpg_name": jpg.name,
                    "variant": variant,
                    "panels": panels,
                }
            )

    index_path = output_dir / "index.html"
    index_path.write_text(render_index(entries))
    print(
        f"wrote {index_path} with {len(entries)} tapestries "
        f"({len(skipped)} (family, style) pairs had no rendered image)"
    )
    if skipped and args.verbose:
        for s in skipped:
            print(f"  skipped: {s}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=REPO_ROOT,
        help="directory holding the rendered <family>-<style>.jpg files "
        "(default: repo root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "public",
        help="directory to write the static site into (default: public/).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="list every skipped (family, style) pair that had no rendered image.",
    )
    args = parser.parse_args()
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
