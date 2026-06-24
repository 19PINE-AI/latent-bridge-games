"""Extract prompts + real slow emissions + per-game eval cells into one JSON the
website consumes. Single source of truth: prompts.py and the eval/trajectory JSONs."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training import prompts as P

OUT = "web-react/src/data/research.json"

# --- 1. System prompts (verbatim) ---
systems = {
    "Atari (default)": P.SYSTEM_PROMPT,
    "MiniGrid": P.MINIGRID_SYSTEM_PROMPT,
    "Highway": P.HIGHWAY_SYSTEM_PROMPT,
    "MetaDrive": P.METADRIVE_SYSTEM_PROMPT,
}

# --- 2. A rendered user prompt + a real slow emission per game, from trajectories ---
# Search patterns per game, first existing match wins (newest schema first).
TRAJ = {
    "MsPacman":   ["results/t_trajectories_v2/MsPacman_seed*.pt", "results/t_trajectories_v2_n16/MsPacman_seed0.pt"],
    "RoadRunner": ["results/t_trajectories_v2/RoadRunner_seed*.pt"],
    "Seaquest":   ["results/t_trajectories_v2/Seaquest_seed*.pt"],
    "Riverraid":  ["results/t_trajectories_v2/Riverraid_seed*.pt"],
    "Qbert":      ["results/t_trajectories_v2/Qbert_seed*.pt"],
    "SpaceInvaders": ["results/t_trajectories_v2/SpaceInvaders_seed*.pt"],
    "Enduro":     ["results/t_trajectories_v2/Enduro_seed*.pt"],
    "MetaDrive":  ["results/t_traj_md_plan/MetaDrive_seed0.pt", "results/t_traj_md_expert/MetaDrive_seed0.pt"],
}

def find_traj(game):
    for pat in TRAJ.get(game, []):
        if os.path.exists(pat):
            return pat
        g = sorted(glob.glob(pat))
        if g:
            return g[0]
    return None

import torch
games = {}
for game, _ in TRAJ.items():
    tp = find_traj(game)
    entry = {"user_prompt": None, "slow_emission": None, "system_key":
             "MetaDrive" if game == "MetaDrive" else "Atari (default)"}
    if tp:
        try:
            trace = torch.load(tp, map_location="cpu", weights_only=False)["trace"]
            # pick a mid-episode step that has both a text_state and a slow emission
            from src.env.atari_wrapper import TextState
            for s in trace[len(trace)//3:]:
                ts = s.get("text_state"); st = s.get("slow_text")
                if isinstance(ts, dict):  # stored as dict -> reconstruct TextState
                    ts = TextState(frame_idx=ts.get("frame_idx", 0), score=ts.get("score", 0),
                                   lives=ts.get("lives", 0), level=ts.get("level", 0),
                                   entities=ts.get("entities", {}))
                if ts is not None and st:
                    # render the user prompt via the real builder
                    try:
                        msgs = P.build_slow_model_messages(game, ts, prior_thought=None)
                        # extract the user text content
                        usr = ""
                        for m in msgs:
                            if m.get("role") == "user":
                                c = m["content"]
                                usr = c if isinstance(c, str) else " ".join(
                                    p.get("text", "") for p in c if isinstance(p, dict))
                        entry["user_prompt"] = usr.strip()[:1400]
                    except Exception as e:
                        entry["user_prompt"] = "(render error: %s)" % e
                    entry["slow_emission"] = st.strip()[:900]
                    break
        except Exception as e:
            entry["error"] = str(e)[:120]
    games[game] = entry

data = {"systems": systems, "games": games}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(data, open(OUT, "w"), indent=1)
print("wrote", OUT)
for g, e in games.items():
    print("  %-14s prompt=%s emission=%s" % (
        g, "Y" if e["user_prompt"] else "—", "Y" if e["slow_emission"] else "—"))
