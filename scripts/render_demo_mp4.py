"""Render a trace directory (frame_XXXXX.npz + events.json) into an MP4 with
the gameplay on the left and the slow model's most-recent text emission as a
side-panel caption on the right.

Used to convert benchmark.py --save-trace-dir output into the demo deliverable.

Usage:
    # single video (one strategy / one episode)
    python scripts/render_demo_mp4.py \
        --trace-dir traces/L_MsPacman_seed0 \
        --out demos/mspacman_L.mp4 --fps 15

    # side-by-side F vs L on the same seed
    python scripts/render_demo_mp4.py \
        --trace-dir traces/F_MsPacman_seed0 traces/L_MsPacman_seed0 \
        --labels F L \
        --out demos/mspacman_F_vs_L.mp4 --fps 15
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Geometry — Atari frames are 210x160; we render at 4x for legibility
FRAME_SCALE = 4
FRAME_W = 160 * FRAME_SCALE  # 640
FRAME_H = 210 * FRAME_SCALE  # 840
CAPTION_W = 480              # text panel
CAPTION_PAD = 16
SCORE_BAR_H = 36

# Try a couple of common DejaVu paths; PIL falls back to default if both fail.
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for p in _FONT_CANDIDATES:
        if bold and "Bold" not in p:
            continue
        if not bold and "Bold" in p:
            continue
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _load_events(trace_dir: Path) -> dict:
    events_path = trace_dir / "events.json"
    return json.loads(events_path.read_text())


def _frame_path(trace_dir: Path, tick: int) -> Path:
    return trace_dir / f"frame_{tick:05d}.npz"


def _render_single_panel(trace_dir: Path, tick: int, ev: dict,
                          strategy_label: str, game: str) -> Image.Image:
    """Render one frame: game (upscaled) over caption + score bar."""
    frame_npz = np.load(_frame_path(trace_dir, tick))["frame"]
    game_img = Image.fromarray(frame_npz.astype(np.uint8))
    game_img = game_img.resize((FRAME_W, FRAME_H), Image.NEAREST)

    panel_w = FRAME_W + CAPTION_W
    panel_h = FRAME_H + SCORE_BAR_H
    out = Image.new("RGB", (panel_w, panel_h), color=(20, 20, 24))
    out.paste(game_img, (0, SCORE_BAR_H))

    draw = ImageDraw.Draw(out)
    title_font = _load_font(20, bold=True)
    body_font = _load_font(18, bold=False)
    label_font = _load_font(16, bold=True)

    # Top score / strategy bar
    bar_text = (f"{strategy_label}  |  {game}  |  tick {ev['tick']:4d}  "
                f"|  action={ev['action']:2d}  |  score={ev['cumulative_reward']:.0f}")
    draw.rectangle([0, 0, panel_w, SCORE_BAR_H], fill=(36, 36, 44))
    draw.text((CAPTION_PAD, 7), bar_text, font=label_font, fill=(220, 220, 240))

    # Right-side caption panel
    cap_x0 = FRAME_W + CAPTION_PAD
    cap_y = SCORE_BAR_H + CAPTION_PAD
    draw.text((cap_x0, cap_y), "slow-model guidance",
              font=title_font, fill=(255, 200, 60))
    cap_y += 30

    txt = (ev.get("latest_slow_text") or "(no emission yet)").strip()
    # Strip <think>...</think> chain-of-thought if present; keep the final emission only
    if "</think>" in txt:
        txt = txt.split("</think>", 1)[-1].strip()
    if "<think>" in txt:
        # only-CoT branch — show a short marker
        txt = "(slow is thinking…)"

    wrap_width = max(20, CAPTION_W // 11)  # ~11 px per char at 18pt
    for paragraph in txt.split("\n"):
        if not paragraph.strip():
            cap_y += 8
            continue
        for line in textwrap.wrap(paragraph, width=wrap_width) or [""]:
            draw.text((cap_x0, cap_y), line, font=body_font, fill=(225, 225, 225))
            cap_y += 22
            if cap_y > panel_h - 30:
                break

    return out


def _render_sidebyside(trace_dirs: list[Path], labels: list[str], tick: int,
                        events_per_dir: list[dict], game: str) -> Image.Image:
    """Render two (or more) traces stacked horizontally for a single tick.

    Each trace contributes one panel from `_render_single_panel`.
    """
    panels = []
    for td, lab, evs in zip(trace_dirs, labels, events_per_dir):
        # Find the event for this tick; clamp to last available if past episode end
        last_ev = evs["events"][-1] if evs["events"] else {"tick": 0, "action": 0,
                                                            "cumulative_reward": 0,
                                                            "latest_slow_text": ""}
        ev = next((e for e in evs["events"] if e["tick"] == tick), last_ev)
        # Also clamp the frame path
        max_tick = max((e["tick"] for e in evs["events"]), default=tick)
        eff_tick = min(tick, max_tick)
        # Re-render with the correct frame for this tick (or the last)
        try:
            p = _render_single_panel(td, eff_tick, ev, lab, game)
        except FileNotFoundError:
            # Fallback: render a blank "episode ended" panel
            p = Image.new("RGB", (FRAME_W + CAPTION_W, FRAME_H + SCORE_BAR_H),
                          color=(0, 0, 0))
            draw = ImageDraw.Draw(p)
            draw.text((20, 20), f"{lab}: episode ended at tick {max_tick}",
                      font=_load_font(20, bold=True), fill=(255, 80, 80))
        panels.append(p)

    total_w = sum(p.width for p in panels) + (len(panels) - 1) * 8
    h = max(p.height for p in panels)
    out = Image.new("RGB", (total_w, h), color=(0, 0, 0))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + 8
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", nargs="+", required=True,
                    help="One or more trace directories. Multiple → side-by-side video.")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Optional labels for each trace dir (default = parse from "
                         "directory name).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--max-ticks", type=int, default=None,
                    help="Cap the output to this many ticks (debug)")
    args = ap.parse_args()

    trace_dirs = [Path(p) for p in args.trace_dir]
    for td in trace_dirs:
        if not td.exists():
            sys.exit(f"trace dir missing: {td}")

    events_per_dir = [_load_events(td) for td in trace_dirs]

    if args.labels:
        labels = args.labels
    else:
        labels = [td.name.split("_", 1)[0] for td in trace_dirs]
    if len(labels) != len(trace_dirs):
        sys.exit("--labels count must match --trace-dir count")

    game = events_per_dir[0]["game"]
    max_tick = max((max(e["tick"] for e in ev["events"]) for ev in events_per_dir),
                   default=0)
    n_ticks = max_tick + 1
    if args.max_ticks:
        n_ticks = min(n_ticks, args.max_ticks)

    # Render frames to a temp dir, then call ffmpeg
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / f".tmp_frames_{out_path.stem}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    print(f"Rendering {n_ticks} ticks × {len(trace_dirs)} traces "
          f"(side-by-side={len(trace_dirs) > 1})...")
    for t in range(n_ticks):
        if len(trace_dirs) == 1:
            ev = events_per_dir[0]["events"]
            this_ev = next((e for e in ev if e["tick"] == t), ev[-1] if ev else
                           {"tick": 0, "action": 0, "cumulative_reward": 0,
                            "latest_slow_text": ""})
            try:
                img = _render_single_panel(trace_dirs[0], t, this_ev, labels[0], game)
            except FileNotFoundError:
                img = Image.new("RGB", (FRAME_W + CAPTION_W, FRAME_H + SCORE_BAR_H),
                                color=(0, 0, 0))
        else:
            img = _render_sidebyside(trace_dirs, labels, t, events_per_dir, game)
        img.save(tmp_dir / f"f_{t:05d}.png", "PNG")
        if (t + 1) % 50 == 0:
            print(f"  rendered {t+1}/{n_ticks}")

    print(f"Encoding to {out_path}...")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(args.fps),
        "-i", str(tmp_dir / "f_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        sys.exit(f"ffmpeg failed: {res.returncode}")

    shutil.rmtree(tmp_dir)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path} ({size_mb:.1f} MB, {n_ticks} ticks at {args.fps} fps)")


if __name__ == "__main__":
    main()
