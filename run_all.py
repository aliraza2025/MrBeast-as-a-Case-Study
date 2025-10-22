# run_all.py — auto-pull IDs from channel handles, then run pipeline

import os, json
import pandas as pd

from channel_ids import resolve_channel_id_from_handle, get_uploads_playlist_id, get_video_ids_from_playlist
from fetch_youtube import fetch_video_stats
from features import rows_to_df
from eda import make_figures
from models import run_models

# ---- Configure channels and size here ----
CHANNEL_HANDLES = ["@MrBeast"]     # you can add more handles later
VIDEOS_PER_CHANNEL = 250           # try 200–300 for prelim

def collect_video_ids(handles):
    all_ids = []
    for h in handles:
        ch_id = resolve_channel_id_from_handle(h)
        upl = get_uploads_playlist_id(ch_id)
        ids = get_video_ids_from_playlist(upl, max_results=VIDEOS_PER_CHANNEL)
        all_ids.extend(ids)
    # unique, keep order
    seen, unique_ids = set(), []
    for vid in all_ids:
        if vid not in seen:
            seen.add(vid)
            unique_ids.append(vid)
    return unique_ids

def main():
    if not os.getenv("YOUTUBE_API_KEY"):
        raise SystemExit("❌ YOUTUBE_API_KEY is not set in this terminal session.")

    video_ids = collect_video_ids(CHANNEL_HANDLES)
    print(f"Collected {len(video_ids)} video IDs from {CHANNEL_HANDLES}")
    if len(video_ids) < 20:
        print(f"⚠️ Only got {len(video_ids)} videos. Consider adding more channels or raising VIDEOS_PER_CHANNEL.")

    # 1) Fetch + tidy
    items = fetch_video_stats(video_ids)
    df = rows_to_df(items)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/youtube_dataset.csv", index=False)
    print(f"✅ Saved dataset with {len(df)} rows to data/youtube_dataset.csv")

    # 2) EDA figs
    make_figures(df)
    print("✅ Saved figures to figs/ (PNG)")

    # 3) Models + metrics (adaptive CV in models.py)
    metrics = run_models(df)
    os.makedirs("figs", exist_ok=True)
    with open("figs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("✅ Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
