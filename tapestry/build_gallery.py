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
HERO_SOURCE = REPO_ROOT / "assets" / "Bayeux-hero.jpg"
HERO_DEPLOY_NAME = "hero.jpg"

VARIANT_PRIORITY = ("-poisson", "-blended", "")


def find_output_jpg(
    images_dir: Path, family_stem: str, style_stem: str
) -> tuple[Path | None, str]:
    """Return (preferred_jpg, variant_label) for a (family, style) pair.

    Priority: ``-poisson`` > ``-blended`` > raw. ``variant_label`` is
    one of ``"poisson"``, ``"blended"``, ``"raw"`` — useful for log
    output, not surfaced to gallery viewers.
    """
    for suffix in VARIANT_PRIORITY:
        candidate = images_dir / f"{family_stem}-{style_stem}{suffix}.jpg"
        if candidate.exists():
            return candidate, suffix.lstrip("-") or "raw"
    return None, ""


# Style names in JSON are lowercase-with-hyphens identifiers; these are
# their display forms. Fall back to ``.title()`` for anything unlisted,
# which handles "hanna-barbera" → "Hanna-Barbera" correctly.
STYLE_DISPLAY = {
    "ukiyo-e": "Ukiyo-e",
}


def style_display(name: str) -> str:
    return STYLE_DISPLAY.get(name, name.title())


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
.hero-banner {
  display: block;
  width: 100%;
  max-height: 440px;
  object-fit: cover;
  object-position: center;
  border-bottom: 1px solid var(--rule);
}
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  /* Crisp system sans for body legibility; serif is reserved for the
     h1 wordmark and panel titles below. */
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, sans-serif;
  font-size: 17px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}
header {
  padding: 3rem 1.5rem 1.5rem;
  text-align: center;
  border-bottom: 1px solid var(--rule);
}
header h1 {
  font-family: Georgia, "Iowan Old Style", "Times New Roman", serif;
  font-size: 3.25rem;
  margin: 0 0 0.5rem;
  letter-spacing: 0.01em;
  font-weight: 600;
}
header > p { margin: 0; color: var(--muted); font-size: 1.05rem; }
header .stats {
  margin-top: 0.5rem;
  color: var(--muted);
  font-size: 0.95rem;
}
.intro {
  max-width: 760px;
  margin: 2.5rem auto 2rem;
  padding: 0 1.5rem;
  font-size: 1.05rem;
  line-height: 1.65;
}
.intro p { margin: 0 0 1.1rem; }
.intro strong { color: var(--ink); }
a.highlight {
  color: var(--accent);
  font-weight: 600;
  text-decoration: underline;
  text-decoration-thickness: 2px;
  text-underline-offset: 3px;
}
a.highlight:hover { background: rgba(122, 46, 32, 0.1); }
.pipeline {
  margin: 0 0 1.5rem;
  padding: 0;
  list-style: none;
}
.pipeline li {
  margin: 0.55rem 0;
  padding: 0.35rem 0.9rem;
  border-left: 3px solid var(--accent);
  background: rgba(122, 46, 32, 0.05);
}
.pipeline li strong { color: var(--accent); font-weight: 600; }
.intro .cta {
  font-style: italic;
  color: var(--muted);
}
.intro .disclaimer {
  padding: 0.75rem 1.1rem;
  background: rgba(122, 46, 32, 0.08);
  border-left: 3px solid var(--accent);
  border-radius: 2px;
  font-size: 0.95rem;
}
.intro .disclaimer strong { color: var(--accent); }
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
  font-size: 1rem;
  line-height: 1.55;
  color: #fff;
}
.sponsors .pitch code {
  background: rgba(255, 255, 255, 0.15);
  padding: 0.1rem 0.35rem;
  border-radius: 3px;
  font-size: 0.9em;
}
.sponsors .pitch a {
  color: #fff;
  font-weight: 600;
  text-decoration: underline;
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
  font-family: Georgia, "Iowan Old Style", "Times New Roman", serif;
  font-size: 1.5rem;
  font-weight: 600;
}
.tapestry h2 .style-descriptor {
  color: var(--muted);
  font-weight: 400;
  font-size: 1.05rem;
  margin-left: 0.35rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, sans-serif;
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
footer .tagline { margin-top: 0.5rem; color: var(--muted); }
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
    style_label = style_display(entry["style_name"])
    return (
        f'<section class="tapestry">\n'
        f"  <h2>{e(entry['family_title'])}"
        f' <span class="style-descriptor">— {e(style_label)} style</span>'
        f"</h2>\n"
        f'  <img class="hero" loading="lazy" '
        f'src="tapestries/{e(entry["jpg_name"])}" '
        f'alt="{e(entry["family_title"])} rendered in {e(style_label)} style">\n'
        f"  <details>\n"
        f"    <summary>{len(entry['panels'])} panels</summary>\n"
        f'    <ol class="panels">\n{panel_html}\n    </ol>\n'
        f"  </details>\n"
        f"</section>"
    )


def render_index(entries: list[dict]) -> str:
    tapestry_html = "\n".join(render_tapestry(entry) for entry in entries)
    n_families = len({entry["family_stem"] for entry in entries})
    n_styles = len({entry["style_name"] for entry in entries})
    if entries:
        stats = (
            f"{len(entries)} tapestries across "
            f"{n_families} famil{'y' if n_families == 1 else 'ies'} and "
            f"{n_styles} style{'' if n_styles == 1 else 's'}."
        )
    else:
        stats = "No tapestries rendered yet."
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
<img class="hero-banner" src="hero.jpg" alt="Detail from the Bayeux Tapestry">
<header>
  <h1>bayeux</h1>
  <p>Family-history tapestries.</p>
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
  <p class="stats">{e(stats)}</p>
</header>
<section class="intro">
  <p>
    A demonstration of the diffusion models at
    <a class="highlight" href="https://nunchaku.dev">nunchaku.dev</a>.
    Built by <strong>Quentin Fennessy</strong> at
    <a class="highlight" href="https://sundai.club">Sundai.club</a>
    on <strong>19 April 2026</strong>. Source and architecture:
    <a class="highlight" href="https://github.com/qfennessy/bayeux">github.com/qfennessy/bayeux</a>.
  </p>
  <p>
    Each tapestry illustrates a fictional family's history in twelve panels.
    Four models cooperate to produce each one:
  </p>
  <ul class="pipeline">
    <li><strong>Claude Opus 4.7</strong> wrote the twelve story paragraphs for each family.</li>
    <li><strong>Gemini 3.1 Flash Lite</strong> turned each paragraph into a concise visual prompt.</li>
    <li><strong>nunchaku-qwen-image</strong> rendered each panel from its prompt.</li>
    <li><strong>Python + Pillow</strong> assembled the twelve panels into a single tapestry.</li>
  </ul>
  <p class="cta">
    Expand the "12 panels" disclosure under each tapestry to read the story
    behind every image.
  </p>
  <p class="disclaimer">
    <strong>All family chronicles are fictional</strong> — invented to
    demonstrate the pipeline. Any resemblance to real people or events is
    coincidental.
  </p>
</section>
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
  <div class="tagline">
    All family chronicles are fictional, created to illustrate the pipeline.
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
    if args.clean and tapestries_out.exists():
        removed = 0
        for p in tapestries_out.iterdir():
            if p.is_file():
                p.unlink()
                removed += 1
        print(
            f"clean: wiped {removed} files from {tapestries_out}; "
            "gallery will be an exact mirror of --images-dir"
        )
    tapestries_out.mkdir(parents=True, exist_ok=True)

    # Copy hero banner (assets/Bayeux-hero.jpg → public/hero.jpg) so it
    # ships with the deployed static site.
    if HERO_SOURCE.exists():
        hero_dest = output_dir / HERO_DEPLOY_NAME
        if (
            not hero_dest.exists()
            or hero_dest.stat().st_mtime < HERO_SOURCE.stat().st_mtime
        ):
            shutil.copy2(HERO_SOURCE, hero_dest)
            print(f"hero: copied {HERO_SOURCE.name} → {hero_dest}")
    else:
        print(
            f"warning: hero image {HERO_SOURCE} not found; "
            "the <img class=\"hero-banner\"> will 404.",
            file=sys.stderr,
        )

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
            # Strip the -poisson / -blended suffix on copy: viewers don't
            # need to know which post-process variant they're looking at,
            # so the public URL is always <family>-<style>.jpg.
            dest_name = f"{family_path.stem}-{style_path.stem}.jpg"
            dest = tapestries_out / dest_name
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
                    "jpg_name": dest_name,
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
    parser.add_argument(
        "--clean",
        action="store_true",
        help="wipe public/tapestries/ before repopulating so the deployed "
        "directory is an exact mirror of what's discoverable in "
        "--images-dir. Use this after deleting images from out/ — without "
        "--clean, orphaned copies linger in public/ and keep getting served.",
    )
    args = parser.parse_args()
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
