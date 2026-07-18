import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.config import RE_DATA_ROOT, LOL_DATA_ROOT

# load red eclipse data
def find_red_eclipse_files():
    return sorted(
        p for p in Path(RE_DATA_ROOT).rglob("*.json")
        if "users" not in p.name and p.name != "sessions.json"
    )

def load_red_eclipse_mouse(file_path):
    with open(file_path) as f:
        game = json.load(f)
    meta = {
        "userId": game["userId"],
        "gameId": game["id"],
        "source_file": Path(file_path).name,
    }
    mouse = [e for e in game["events"] if e.get("type") == "MouseEvent"]
    df = pd.DataFrame(mouse)
    if not df.empty:
        df = df[["dx", "dy", "time"]]
    return meta, df

# load lol keylogger data
def find_lol_keylogger_files():
    return sorted(Path(LOL_DATA_ROOT).glob("sessions/**/*-keylogger-new.txt"))

def parse_lol_keylogger(file_path, max_events=None, max_minutes=None):
    rows = []
    t0_ms = None
    with open(file_path, encoding="utf-8", errors="replace") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith(("Subject", "File ", "Version", "Released", "Elapsed")):
                continue
            parts = line.split("\t")
            if len(parts) < 4 or parts[1].strip() != "Moved":
                continue
            ts_str = parts[0]
            try:
                date_part, time_part = ts_str.split("T")
                if "." in time_part:
                    hms, frac = time_part.rsplit(".", 1)
                    extra_ms = int(frac)
                else:
                    hms, extra_ms = time_part, 0
                base = datetime.strptime(f"{date_part}T{hms}", "%Y-%m-%dT%H:%M:%S")
                time_ms = int(base.timestamp() * 1000) + extra_ms
                if t0_ms is None:
                    t0_ms = time_ms
                if max_minutes and (time_ms - t0_ms) > max_minutes * 60 * 1000:
                    break
                rows.append({"x": int(parts[2]), "y": int(parts[3]), "time_ms": time_ms})
                if max_events and len(rows) >= max_events:
                    break
            except ValueError:
                continue

    if len(rows) < 2:
        return None

    df = pd.DataFrame(rows).sort_values("time_ms")
    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()
    df["time"] = df["time_ms"] - df["time_ms"].iloc[0]
    return df.iloc[1:][["dx", "dy", "time"]].reset_index(drop=True)
