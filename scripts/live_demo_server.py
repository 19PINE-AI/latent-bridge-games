"""Live demo server — Flask + Server-Sent Events.

Two modes:
  - REPLAY (default): replays a saved trace dir at 15 Hz over SSE. No GPU
    required at serve time.
  - LIVE (--live): runs a fresh benchmark episode and streams frames as they're
    produced. Requires the GPU.

The web/ directory's index.html is served at /, and the live-playback UI is at
/live. Streams from /stream?game=...&strategy=...&seed=... (REPLAY mode looks for
traces/demo/<strategy>_<game>_seed<seed>; LIVE mode would launch a benchmark).

Run:
    python scripts/live_demo_server.py --host 0.0.0.0 --port 8000
    open http://localhost:8000/
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from flask import Flask, Response, request, send_file, send_from_directory
except ImportError:
    print("Flask is required: pip install flask", file=sys.stderr)
    raise

REPO_ROOT = Path(__file__).parent.parent
# Prefer the React build (web-dist) when present; fall back to the static HTML.
WEB_DIR = (REPO_ROOT / "web-dist") if (REPO_ROOT / "web-dist" / "index.html").exists() else (REPO_ROOT / "web")
DEMOS_DIR = REPO_ROOT / "demos"
TRACES_DIR = REPO_ROOT / "traces" / "demo"


app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")


@app.route("/")
def index():
    return send_from_directory(str(WEB_DIR), "index.html")


@app.route("/demos/<path:fn>")
def demos(fn):
    return send_from_directory(str(DEMOS_DIR), fn)


@app.route("/live")
def live_page():
    """Minimal live-playback UI."""
    return Response(LIVE_HTML, mimetype="text/html")


@app.route("/stream")
def stream():
    """Server-Sent Events stream of per-tick {frame_jpeg_b64, score, slow_text}.

    Query params:
      game: MsPacman | Seaquest | SpaceInvaders
      strategy: F | L | T
      seed: int (default 0)
      fps: int (default 15) — replay frame rate
    """
    game = request.args.get("game", "MsPacman")
    strategy = request.args.get("strategy", "L")
    seed = int(request.args.get("seed", "0"))
    fps = max(1, int(request.args.get("fps", "15")))

    trace = TRACES_DIR / f"{strategy}_{game}_seed{seed}"
    if not trace.exists():
        return Response(
            f"data: {json.dumps({'error': f'no trace at {trace}'})}\n\n",
            mimetype="text/event-stream",
        )

    events_path = trace / "events.json"
    events = json.loads(events_path.read_text())["events"]

    def gen():
        dt = 1.0 / fps
        yield "retry: 1000\n\n"
        for ev in events:
            t = ev["tick"]
            frame_path = trace / f"frame_{t:05d}.npz"
            if not frame_path.exists():
                continue
            frame = np.load(frame_path)["frame"]
            img = Image.fromarray(frame.astype(np.uint8))
            # Upscale 3x for screen visibility
            img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            jpeg_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            slow_text = ev.get("latest_slow_text") or ""
            if "</think>" in slow_text:
                slow_text = slow_text.split("</think>", 1)[-1].strip()
            elif "<think>" in slow_text:
                slow_text = "(slow is thinking…)"

            payload = {
                "tick": t,
                "action": ev["action"],
                "score": ev["cumulative_reward"],
                "slow_text": slow_text,
                "frame_jpeg_b64": jpeg_b64,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(dt)
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/healthz")
def healthz():
    n_traces = len(list(TRACES_DIR.glob("*"))) if TRACES_DIR.exists() else 0
    return {"ok": True, "n_traces": n_traces, "demos_dir": str(DEMOS_DIR)}


LIVE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Live latent bridge playback</title>
<style>
  body { background:#0e0f13; color:#e6e7eb; font:14px/1.4 -apple-system,sans-serif;
         margin: 24px auto; max-width: 1100px; padding: 0 16px; }
  h1 { font-size: 22px; margin: 0 0 12px; }
  .controls { display: flex; gap: 12px; margin: 12px 0; align-items: center; }
  select, button { background:#1d2029; color:#e6e7eb; border:1px solid #2a2e3a;
                    padding: 6px 10px; border-radius: 6px; font-size: 14px; }
  button { cursor: pointer; }
  .stage { display: grid; grid-template-columns: 480px 1fr; gap: 16px;
           background:#16181f; padding: 16px; border-radius: 10px;
           border: 1px solid #232733; }
  .stage img { width: 480px; image-rendering: pixelated; background:#000; }
  .meta { font-family: ui-monospace, monospace; }
  .meta h3 { margin: 0 0 4px; color: #ffb84d; font-size: 14px; }
  .slow-text { background:#0e0f13; padding: 10px 12px; border-radius: 6px;
               margin: 8px 0; min-height: 80px; white-space: pre-wrap; }
  .nav { color: #9094a4; }
  .nav a { color:#6fb1ff; }
</style></head><body>
<h1>Live latent bridge playback</h1>
<p class="nav"><a href="/">← back to demos page</a></p>
<div class="controls">
  Game:
  <select id="game">
    <option>MsPacman</option><option>Seaquest</option><option>SpaceInvaders</option>
  </select>
  Strategy:
  <select id="strategy">
    <option value="L">L (latent bridge)</option>
    <option value="F">F (fast only)</option>
  </select>
  Seed: <select id="seed"><option>0</option></select>
  FPS: <select id="fps"><option>15</option><option>30</option><option>60</option></select>
  <button id="play">▶ Play</button>
  <button id="stop">■ Stop</button>
</div>
<div class="stage">
  <img id="frame" alt="game frame">
  <div class="meta">
    <h3>state</h3>
    <div id="state">tick — | action — | score —</div>
    <h3>slow-model guidance</h3>
    <div class="slow-text" id="slow"></div>
  </div>
</div>
<script>
let es = null;
const $ = (id) => document.getElementById(id);
function stop() { if (es) { es.close(); es = null; } }
function play() {
  stop();
  const g = $('game').value, s = $('strategy').value;
  const seed = $('seed').value, fps = $('fps').value;
  const url = `/stream?game=${g}&strategy=${s}&seed=${seed}&fps=${fps}`;
  es = new EventSource(url);
  es.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if (d.error) { $('state').textContent = 'error: ' + d.error; stop(); return; }
    if (d.done) { stop(); return; }
    $('frame').src = 'data:image/jpeg;base64,' + d.frame_jpeg_b64;
    $('state').textContent =
      `tick ${d.tick} | action ${d.action} | score ${d.score}`;
    $('slow').textContent = d.slow_text || '(no emission yet)';
  };
  es.onerror = () => { stop(); };
}
$('play').onclick = play;
$('stop').onclick = stop;
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print(f"Serving on http://{args.host}:{args.port}/")
    print(f"  static demos:   {DEMOS_DIR}")
    print(f"  traces source:  {TRACES_DIR}")
    print(f"  /         — demos site")
    print(f"  /live     — live playback UI")
    print(f"  /stream   — SSE endpoint")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
