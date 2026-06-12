"""Build a single narrated demo video combining all 7 game side-by-side comparisons.

Output:
  demos/combined_narrated.mp4 — single ~5 minute MP4 with:
    - Intro title card + narration
    - Per-game title card (game name + F/T/L scores + headline metric)
    - The side-by-side F-vs-L video clip
    - Narration explaining the result
    - Outro card

Pipeline:
  1. Generate narration .mp3 for each block — ElevenLabs if ELEVENLABS_API_KEY
     is set (model eleven_v3 by default), else gTTS
  2. Generate title-card PNGs (PIL) and convert each to a short video clip
  3. Pad each demo clip's audio with the narration mp3
  4. Concatenate all clips with ffmpeg

Run:
  ELEVENLABS_API_KEY=... python scripts/build_combined_demo.py --out demos/combined_narrated.mp4
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont
try:
    from gtts import gTTS  # type: ignore
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

# ElevenLabs narration. Key comes from the environment only — this repo is public.
ELEVEN_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVEN_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_v3")
# Daniel — "Steady Broadcaster", informative/educational premade voice.
ELEVEN_VOICE = os.environ.get("ELEVENLABS_VOICE", "onwK4e9ZLuTAKqWW03F9")


# Canvas sized for the 3-panel F/T/L clips (~3376x876). All clips/cards are scaled
# to fit inside this box with aspect ratio preserved (letterbox/pillarbox), never stretched.
VIDEO_W = 3376
VIDEO_H = 876
FPS = 15
# aspect-preserving fit-and-pad filter
FIT = (f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=decrease:flags=neighbor,"
       f"pad={VIDEO_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2:color=0x0e0f13,format=yuv420p")

# Per-game segment configuration. The order is chosen to tell a story:
# headline wins first, then the categorical exception, then the diagnosis games.
# Each game segment uses the 3-way F / T / L clip so the text bridge is visible too.
SEGMENTS = [
    {
        "id": "intro",
        "kind": "card",
        "title": "Latent Bridge",
        "subtitle": "Fast/slow model coupling for real-time agents",
        "body": [
            "Frozen 9B reactive model + frozen 8B reasoning model",
            "Coupled via a 33M-param latent token bridge",
            "7 Atari games + a driving sim (MetaDrive)",
        ],
        "narration": (
            "This is the Latent Bridge. "
            "A fast reactive model acts every frame; a slow reasoning model thinks once a second. "
            "We compare three ways to couple them: fast-only, a text bridge that passes the slow "
            "model's words, and a learned latent bridge that passes its continuous residuals. "
            "Each clip shows fast, text, and latent side by side."
        ),
        "card_duration": 6,
    },
    {
        "id": "roadrunner",
        "kind": "game",
        "title": "Road Runner",
        "subtitle": "F = 0    T = 608    L = 967",
        "headline": "Fast alone scores zero; the latent bridge unlocks the policy",
        "video": "demos/roadrunner_F_T_L.mp4",
        "narration": (
            "Road Runner: our cleanest win. "
            "The fast model alone scores zero — it has the reflexes but won't commit a direction. "
            "The text bridge reaches 608; the latent bridge, 967. "
            "Watch all three play side by side."
        ),
    },
    {
        "id": "mspacman",
        "kind": "game",
        "title": "Ms. Pac-Man",
        "subtitle": "F = 256    T = 408    L = 628",
        "headline": "Latent beats text by 54% on the original headline game",
        "video": "demos/mspacman_F_T_L.mp4",
        "narration": (
            "Ms. Pac-Man, our original headline. "
            "Fast 256, text 408, latent 628. "
            "The latent channel carries richer joint state per emission — "
            "ghost and pellet positions together — than text serializes compactly."
        ),
    },
    {
        "id": "riverraid",
        "kind": "game",
        "title": "River Raid",
        "subtitle": "Robust Stage A: F = 1033    T = 337    L = 612",
        "headline": "Largest L-over-T gap after fixing Stage A OOD brittleness",
        "video": "demos/riverraid_F_T_L.mp4",
        "narration": (
            "River Raid: the largest latent-over-text gap in the sweep, 82 percent. "
            "But notice fast-only still leads at 1033 — the bridges help relative to each other, "
            "not over fast here. The recovery came from fixing Stage A out-of-distribution brittleness."
        ),
    },
    {
        "id": "seaquest",
        "kind": "game",
        "title": "Seaquest",
        "subtitle": "F = 42    T = 63    L = 80",
        "headline": "Both bridges beat fast; latent leads",
        "video": "demos/seaquest_F_T_L.mp4",
        "narration": (
            "Seaquest: both bridges beat fast-only, and latent leads at 80 versus 63. "
            "The latent policy locks into a stable surfacing and kill pattern."
        ),
    },
    {
        "id": "qbert",
        "kind": "game",
        "title": "Q*bert — Text Wins",
        "subtitle": "F = 25    T = 125    L = 50",
        "headline": "The counter-example: text beats latent here",
        "video": "demos/qbert_F_T_L.mp4",
        "narration": (
            "Q*bert is the honest counter-example. "
            "Its guidance is categorical — jump up-right to a target tile — "
            "which fits losslessly into text. Here text wins: 125 versus latent's 50. "
            "The bridge is not always better; it depends on the task."
        ),
    },
    {
        "id": "spaceinvaders",
        "kind": "game",
        "title": "Space Invaders — The Diagnosis",
        "subtitle": "F = 107    T = 18    L = 15",
        "headline": "Fast dominates; both bridges fail — a controlled negative",
        "video": "demos/spaceinvaders_F_T_L.mp4",
        "narration": (
            "Space Invaders: fast-only dominates at 107, and both bridges sit near 18. "
            "Slow reasoning does not help this reactive task, so the bridge has nothing useful to carry. "
            "This negative is a clue to the general rule."
        ),
    },
    {
        "id": "enduro",
        "kind": "game",
        "title": "Enduro",
        "subtitle": "Robust: F = 1    T = 5    L = 8",
        "headline": "Small absolute scores; reported for completeness",
        "video": "demos/enduro_F_T_L.mp4",
        "narration": (
            "Enduro: scores are tiny because the expert was weak, "
            "so we report it for completeness rather than as a confident win."
        ),
    },
    {
        "id": "metadrive",
        "kind": "game",
        "title": "MetaDrive — Beyond Atari",
        "subtitle": "Driving sim:  F = 88    T = 85    L = 85",
        "headline": "A non-Atari domain, and a controlled negative",
        "video": "demos/metadrive_F_T_L.mp4",
        "narration": (
            "We also leave Atari for MetaDrive, a real-time driving simulator. "
            "Even on a route that requires planning, slow reasoning never beats the fast reactive policy — "
            "fast 88, text and latent both 85. "
            "Driving is a tight perception-action loop, so the bridge stays inert. "
            "This controlled negative is the key to the general rule."
        ),
    },
    {
        "id": "predictor",
        "kind": "card",
        "title": "The Predictor",
        "subtitle": "L − F tracks T − F at  r = 0.92",
        "body": [
            "The latent bridge helps if and only if slow reasoning helps the task (T > F)",
            "r = 0.92 over 8 best-variant tasks; r = 0.94 over all 16 game/variant cells",
            "Bridge pays off where the bottleneck is deliberation, not perception-action",
        ],
        "narration": (
            "Pulling it together: across seven Atari games and MetaDrive, "
            "the latent bridge's benefit tracks the text bridge's benefit at correlation zero point nine two. "
            "The bridge helps if and only if slow reasoning beats fast reaction on the task. "
            "Whether a latent bridge is worth it is a property of the task, not the channel."
        ),
        "card_duration": 9,
    },
    {
        "id": "bridgereplace",
        "kind": "card",
        "title": "Is the latent real?",
        "subtitle": "Bridge-replacement control on every game",
        "body": [
            "Replace the trained latent with zeros or random vectors at matched norm",
            "Trained >> controls only where slow helps: RoadRunner 967 vs 0, Seaquest, MsPacman",
            "Inert or harmful where it doesn't — same as MetaDrive",
        ],
        "narration": (
            "To prove the latent carries real learned content, we replace it with zeros or random vectors. "
            "On the games where slow reasoning helps, the trained latent far exceeds both controls — "
            "Road Runner drops from 967 to zero when the latent is removed. "
            "Where slow reasoning doesn't help, the trained latent is no better, or even harmful. "
            "How much of the latent is learned is itself predicted by whether slow beats fast."
        ),
        "card_duration": 9,
    },
    {
        "id": "outro",
        "kind": "card",
        "title": "Summary",
        "subtitle": "7 Atari games + MetaDrive · the T > F predictor",
        "body": [
            "Latent helps iff slow reasoning helps the task (T > F), r = 0.92",
            "Bridge-replacement control: learned content exactly where it helps",
            "MetaDrive: the controlled negative; Q*bert: text wins (categorical)",
            "Stage A OOD brittleness diagnosed and fixed; bandwidth thesis retired",
        ],
        "narration": (
            "In summary: the latent bridge helps if and only if slow reasoning helps the task, "
            "shown across Atari and a driving domain, with a bridge-replacement control and a clean negative. "
            "Code, paper, and interactive replays at github dot com slash bojieli slash latent dash bridge dash games."
        ),
        "card_duration": 9,
    },
]


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else None,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_card_png(seg: dict, out_path: Path) -> None:
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), color=(14, 15, 19))
    draw = ImageDraw.Draw(img)

    title_font = _font(96, bold=True)
    subtitle_font = _font(52, bold=False)
    body_font = _font(40, bold=False)
    accent_font = _font(48, bold=True)

    title = seg["title"]
    subtitle = seg.get("subtitle", "")
    headline = seg.get("headline", "")
    body = seg.get("body", [])

    # Vertically center the block of text
    y = 140
    draw.text((VIDEO_W // 2, y), title, font=title_font, fill=(255, 184, 77),
              anchor="mm")
    y += 110
    if subtitle:
        draw.text((VIDEO_W // 2, y), subtitle, font=subtitle_font,
                  fill=(225, 225, 235), anchor="mm")
        y += 80
    if headline:
        draw.text((VIDEO_W // 2, y), headline, font=accent_font,
                  fill=(95, 217, 145), anchor="mm")
        y += 70
    if body:
        for line in body:
            draw.text((VIDEO_W // 2, y), line, font=body_font,
                      fill=(186, 186, 200), anchor="mm")
            y += 56

    img.save(out_path, "PNG")


def generate_narration_mp3(text: str, out_path: Path) -> None:
    if ELEVEN_KEY:
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}",
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": ELEVEN_KEY},
            json={"text": text, "model_id": ELEVEN_MODEL},
            timeout=120,
        )
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return
    if not HAS_GTTS:
        # Fallback: empty audio
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=cl=stereo:r=24000",
             "-t", "3", "-q:a", "9", str(out_path)],
            check=True, capture_output=True,
        )
        return
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(str(out_path))


def make_card_video(card_png: Path, audio_mp3: Path, out_mp4: Path,
                     min_duration: float) -> None:
    """Combine a static PNG with an audio track into an MP4 clip."""
    # Get audio duration
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_mp3)],
        capture_output=True, text=True,
    )
    try:
        audio_dur = float(res.stdout.strip() or "0")
    except ValueError:
        audio_dur = 3.0
    duration = max(min_duration, audio_dur + 0.5)
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(card_png),
        "-i", str(audio_mp3),
        "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-vf", FIT,
        "-r", str(FPS),
        "-t", f"{duration:.2f}",
        "-shortest", str(out_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def make_game_video(video_in: Path, audio_mp3: Path, out_mp4: Path) -> None:
    """Re-encode a gameplay clip and overlay narration audio."""
    # Use the gameplay video for the visual, narration mp3 as the audio.
    # If narration is shorter than the video, the video plays out silently after.
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_in),
        "-i", str(audio_mp3),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", FIT,
        "-r", str(FPS),
        "-c:a", "aac", "-b:a", "128k",
        "-shortest", str(out_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _probe_dur(path: Path) -> float:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    try:
        return float(res.stdout.strip() or "0")
    except ValueError:
        return 0.0


def _srt_ts(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(entries: list[tuple[float, float, str]], out_path: Path) -> None:
    """entries: list of (start_sec, end_sec, text)."""
    lines = []
    for i, (st, en, text) in enumerate(entries, 1):
        lines.append(str(i))
        lines.append(f"{_srt_ts(st)} --> {_srt_ts(en)}")
        lines.append(text.strip())
        lines.append("")
    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demos/combined_narrated.mp4")
    ap.add_argument("--workdir", default="demos/_combined_tmp")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    # (timeline_clip_index_start, segment) so we can place subtitles on the right span
    seg_spans: list[tuple[Path, str]] = []  # (clip that carries the narration, narration text)
    for i, seg in enumerate(SEGMENTS):
        seg_id = seg["id"]
        narr_mp3 = workdir / f"{i:02d}_{seg_id}.mp3"
        print(f"[{i:02d}/{len(SEGMENTS)-1}] {seg_id}: generating narration")
        generate_narration_mp3(seg["narration"], narr_mp3)

        if seg["kind"] == "card":
            card_png = workdir / f"{i:02d}_{seg_id}_card.png"
            render_card_png(seg, card_png)
            clip_mp4 = workdir / f"{i:02d}_{seg_id}.mp4"
            make_card_video(card_png, narr_mp3, clip_mp4,
                             min_duration=seg.get("card_duration", 5))
            parts.append(clip_mp4)
            seg_spans.append((clip_mp4, seg["narration"]))
        else:
            # Title card BEFORE the gameplay
            card_png = workdir / f"{i:02d}_{seg_id}_card.png"
            render_card_png(seg, card_png)
            title_clip = workdir / f"{i:02d}_{seg_id}_title.mp4"
            # Short silent title card (3s) — keeps the narration concentrated on gameplay
            subprocess.run(
                ["ffmpeg", "-y", "-loop", "1", "-i", str(card_png),
                 "-f", "lavfi", "-i", "anullsrc=cl=stereo:r=24000",
                 "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                 "-vf", FIT,
                 "-r", str(FPS), "-c:a", "aac", "-b:a", "128k",
                 "-t", "3", "-shortest", str(title_clip)],
                check=True, capture_output=True,
            )
            parts.append(title_clip)
            # Then the gameplay with narration overlaid as audio
            game_clip = workdir / f"{i:02d}_{seg_id}_play.mp4"
            make_game_video(Path(seg["video"]), narr_mp3, game_clip)
            parts.append(game_clip)
            seg_spans.append((game_clip, seg["narration"]))

    # Concatenate
    concat_list = workdir / "concat.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Concatenating {len(parts)} clips → {out_path}")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c", "copy", str(out_path)],
        check=True, capture_output=True,
    )
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # ---- Build a matching SRT from real per-clip durations ----
    narr_for = {p: txt for p, txt in seg_spans}
    entries: list[tuple[float, float, str]] = []
    t0 = 0.0
    for p in parts:
        d = _probe_dur(p)
        if p in narr_for:
            # subtitle spans this clip (clamped a hair inside its bounds)
            entries.append((t0, t0 + max(d - 0.1, 0.5), narr_for[p]))
        t0 += d
    srt_path = out_path.with_suffix(".srt")
    write_srt(entries, srt_path)
    print(f"Wrote {srt_path} ({len(entries)} cues)")

    if not args.keep_tmp:
        import shutil
        shutil.rmtree(workdir)
        print(f"Removed {workdir}")


if __name__ == "__main__":
    main()
