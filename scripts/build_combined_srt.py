"""Generate an SRT subtitle file for the combined narrated demo.

Reads the segment definitions from build_combined_demo.SEGMENTS, computes
timing from the actual narration MP3 durations (regenerated for measurement)
plus the 3-second per-game title cards, and emits demos/combined_narrated.srt.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from gtts import gTTS

import sys
sys.path.insert(0, "scripts")
from build_combined_demo import SEGMENTS, FPS


def _audio_duration(path: Path) -> float:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 3.0


def _fmt_ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _wrap_text(text: str, max_per_line: int = 70) -> str:
    """Wrap text by sentence + word."""
    import textwrap
    lines = []
    for sent in text.replace(". ", ".|").split("|"):
        for line in textwrap.wrap(sent, width=max_per_line):
            lines.append(line)
    return "\n".join(lines[:3])  # SRT recommends max 3 lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demos/combined_narrated.srt")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="srt_"))
    cur = 0.0
    entries = []

    for i, seg in enumerate(SEGMENTS):
        narr_mp3 = tmp / f"{i:02d}.mp3"
        gTTS(text=seg["narration"], lang="en", slow=False).save(str(narr_mp3))
        narr_dur = _audio_duration(narr_mp3)

        if seg["kind"] == "card":
            # Card duration = max(card_duration, narr_dur + 0.5)
            seg_dur = max(seg.get("card_duration", 5), narr_dur + 0.5)
            start = cur
            end = cur + min(narr_dur, seg_dur)
            entries.append((start, end, _wrap_text(seg["narration"])))
            cur += seg_dur
        else:
            # Game segment: 3s silent title card + game video (audio overlaid is narration)
            cur += 3.0  # silent title
            # Game video duration
            video_dur = _audio_duration(Path(seg["video"]))  # close enough for our needs
            start = cur
            end = cur + min(narr_dur, video_dur)
            entries.append((start, end, _wrap_text(seg["narration"])))
            cur += video_dur

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, (start, end, text) in enumerate(entries, 1):
            f.write(f"{i}\n{_fmt_ts(start)} --> {_fmt_ts(end)}\n{text}\n\n")

    shutil.rmtree(tmp)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
