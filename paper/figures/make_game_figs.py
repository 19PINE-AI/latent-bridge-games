#!/usr/bin/env python3
"""Build two paper figures from demos/*_F_T_L.mp4:
  fig_games_grid.pdf     -- 2x4 orientation montage of the 8 evaluated domains
  fig_mspacman_compare.pdf -- F|T|L comparison still for MsPacman (game panels only)
Frames are pulled with ffmpeg; layout via matplotlib so output matches the
existing PDF figures.
"""
import os, subprocess, json
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/ubuntu/latent-bridge-games"
DEMOS = os.path.join(REPO, "demos")
TMP = "/tmp/claude-1000/-home-ubuntu-latent-bridge-games/9190ba6e-7be4-4edc-8655-0c211f5e5c74/scratchpad/frames"
OUT = os.path.join(REPO, "paper", "figures")
os.makedirs(TMP, exist_ok=True)

def dur(path):
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                        "-of","default=nk=1:nw=1", path], capture_output=True, text=True)
    try: return float(r.stdout.strip())
    except: return 30.0

def grab(mp4, ts, dst):
    subprocess.run(["ffmpeg","-y","-ss",f"{ts:.2f}","-i",mp4,"-frames:v","1",dst],
                   capture_output=True)
    return dst

# (display name, demo basename, frac-of-duration, atari_crop_or_metadrive)
GAMES = [
    ("Ms. Pac-Man",    "mspacman",      0.50, "atari"),
    ("Road Runner",    "roadrunner",    0.55, "atari"),
    ("Seaquest",       "seaquest",      0.50, "atari"),
    ("Q*bert",         "qbert",         0.55, "atari"),
    ("River Raid",     "riverraid",     0.50, "atari"),
    ("Space Invaders", "spaceinvaders", 0.45, "atari"),
    ("Enduro",         "enduro",        0.50, "atari"),
    ("MetaDrive (driving)", "metadrive", 0.50, "metadrive"),
]

# crop within the LEFT (F) panel of the F_T_L composite
ATARI_CROP = (4, 38, 640, 872)        # x0,y0,x1,y1 -> game playfield + score
META_CROP  = (250, 36, 740, 872)

def left_panel(img, n_panels=3):
    w, h = img.size
    return img.crop((0, 0, w // n_panels, h))

def game_thumb(name, base, frac, kind):
    mp4 = os.path.join(DEMOS, f"{base}_F_T_L.mp4")
    ts = max(1.0, dur(mp4) * frac)
    f = grab(mp4, ts, os.path.join(TMP, f"thumb_{base}.png"))
    img = Image.open(f).convert("RGB")
    panel = left_panel(img)
    crop = ATARI_CROP if kind == "atari" else META_CROP
    return panel.crop(crop)

# ---------- Figure A: 2x4 orientation grid ----------
fig, axes = plt.subplots(2, 4, figsize=(11, 5.0))
for ax, (name, base, frac, kind) in zip(axes.ravel(), GAMES):
    thumb = game_thumb(name, base, frac, kind)
    ax.imshow(thumb)
    ax.set_title(name, fontsize=11, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#888"); s.set_linewidth(0.6)
plt.tight_layout(pad=0.6, w_pad=0.8, h_pad=1.2)
for ext in ("pdf","png"):
    fig.savefig(os.path.join(OUT, f"fig_games_grid.{ext}"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig_games_grid")

# ---------- Figure B: MsPacman F|T|L comparison (game panels only, keep top label bar) ----------
mp4 = os.path.join(DEMOS, "mspacman_F_T_L.mp4")
ts = max(1.0, dur(mp4) * 0.50)
f = grab(mp4, ts, os.path.join(TMP, "mspac_compare.png"))
comp = Image.open(f).convert("RGB")
W, H = comp.size
pw = W // 3
# keep the top bar (F/T/L + score) -> y0=0; crop out the guidance text box
PANEL_CROP = (4, 0, 640, H)
panels = [comp.crop((i*pw, 0, (i+1)*pw, H)).crop(PANEL_CROP) for i in range(3)]
figB, axesB = plt.subplots(1, 3, figsize=(9.5, 4.4))
for ax, p in zip(axesB, panels):
    ax.imshow(p); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
plt.tight_layout(pad=0.3, w_pad=0.4)
for ext in ("pdf","png"):
    figB.savefig(os.path.join(OUT, f"fig_mspacman_compare.{ext}"), dpi=150, bbox_inches="tight")
plt.close(figB)
print("wrote fig_mspacman_compare")
