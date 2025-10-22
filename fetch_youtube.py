# fetch_youtube.py
import os, time, requests

API_KEY = os.getenv("YOUTUBE_API_KEY")
BASE = "https://www.googleapis.com/youtube/v3"

def fetch_video_stats(video_ids, batch_size=50, throttle=0.05):
    """
    Fetch stats for many videos, batching <=50 IDs per request (API limit).
    throttle: small sleep between calls to be polite with quota.
    """
    if not API_KEY:
        raise RuntimeError("Set YOUTUBE_API_KEY environment variable first.")
    all_items = []
    for i in range(0, len(video_ids), batch_size):
        chunk = video_ids[i:i+batch_size]
        params = {
            "part": "snippet,contentDetails,statistics",
            "id": ",".join(chunk),
            "key": API_KEY,
            "fields": (
                "items(id,"
                "snippet(title,publishedAt),"
                "contentDetails(duration),"
                "statistics(viewCount,likeCount,commentCount))"
            ),
            # maxResults is ignored for videos.list; the ID list drives the result count
        }
        r = requests.get(f"{BASE}/videos", params=params, timeout=30)
        r.raise_for_status()
        all_items.extend(r.json().get("items", []))
        time.sleep(throttle)  # tiny pause to avoid spiky quota usage
    return all_items
