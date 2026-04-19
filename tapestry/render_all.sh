#!/usr/bin/env bash
# Render every (family, style) combination at 12 panels with --full-blend.
# Skips any combo whose assembled <family>-<style>.jpg already exists in
# out/. Safe to re-run: the per-panel cache makes it resumable, so a
# rate-limit failure on combo N just means you restart and it picks up
# where it left off.

set -u
cd "$(dirname "$0")/.."

# Load API keys.
set -a
# shellcheck disable=SC1091
source tapestry/.env
set +a

mkdir -p out

total=0
done_count=0
skipped=0
failed=()

families=(tapestry/*.json)
styles=(tapestry/styles/*.json)

for fam in "${families[@]}"; do
    for sty in "${styles[@]}"; do
        total=$((total + 1))
        fam_stem=$(basename "$fam" .json)
        sty_stem=$(basename "$sty" .json)
        out_file="out/${fam_stem}-${sty_stem}.jpg"
        if [ -f "$out_file" ]; then
            printf "[SKIP %2d/30] %s × %s\n" "$total" "$fam_stem" "$sty_stem"
            skipped=$((skipped + 1))
            continue
        fi
        printf "\n============================================\n"
        printf "[RENDER %2d/30] %s × %s\n" "$total" "$fam_stem" "$sty_stem"
        printf "============================================\n"
        if python -u tapestry/build_tapestry.py "$fam" --style "$sty" \
            --limit 12 --full-blend --out-dir out; then
            done_count=$((done_count + 1))
        else
            printf "[FAIL %2d/30] %s × %s (will be retried on next run)\n" \
                "$total" "$fam_stem" "$sty_stem"
            failed+=("${fam_stem}-${sty_stem}")
        fi
        sleep 2
    done
done

printf "\n=== render_all summary ===\n"
printf "total combos:        %d\n" "$total"
printf "skipped (existing):  %d\n" "$skipped"
printf "rendered this run:   %d\n" "$done_count"
printf "failed this run:     %d\n" "${#failed[@]}"
if [ "${#failed[@]}" -gt 0 ]; then
    printf "  failures:\n"
    for f in "${failed[@]}"; do
        printf "    %s\n" "$f"
    done
fi
printf "assembled now in out/: %d\n" "$(ls out/*.jpg 2>/dev/null | grep -v -- '-poisson' | wc -l | tr -d ' ')"
