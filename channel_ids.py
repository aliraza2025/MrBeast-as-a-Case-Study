# channel_ids.py
import os, requests

API_KEY = os.getenv("YOUTUBE_API_KEY")
BASE = "https://www.googleapis.com/youtube/v3"

def resolve_channel_id_from_handle(handle: str) -> str:
    """
    Works for handles like '@MrBeast'. Uses search.list to find the channelId.
    """
    if handle.startswith("@"):
        q = handle
    else:
        q = f"@{handle}"
    params = {
        "part": "snippet",
        "type": "channel",
        "q": q,
        "maxResults": 1,
        "key": API_KEY
    }
    r = requests.get(f"{BASE}/search", params=params, timeout=30)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        raise ValueError(f"No channel found for handle {handle}")
    return items[0]["id"]["channelId"]

def get_uploads_playlist_id(channel_id: str) -> str:
    """
    channels.list to fetch the uploads playlist id.
    """
    params = {"part": "contentDetails", "id": channel_id, "key": API_KEY}
    r = requests.get(f"{BASE}/channels", params=params, timeout=30)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        raise ValueError(f"No channel contentDetails for {channel_id}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

def get_video_ids_from_playlist(playlist_id: str, max_results: int = 300) -> list[str]:
    """
    Walks playlistItems.list to collect up to max_results video IDs.
    """
    ids, page_token = [], None
    while len(ids) < max_results:
        params = {
            "part": "contentDetails",
            "playlistId": playlist_id,
            "maxResults": 50,
            "key": API_KEY
        }
        if page_token:
            params["pageToken"] = page_token
        r = requests.get(f"{BASE}/playlistItems", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for it in data.get("items", []):
            ids.append(it["contentDetails"]["videoId"])
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return ids[:max_results]
