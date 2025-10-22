import pandas as pd, numpy as np, isodate

def rows_to_df(items):
    rows = []
    for it in items:
        s, c, st = it["snippet"], it["contentDetails"], it["statistics"]
        dur_s = isodate.parse_duration(c["duration"]).total_seconds()
        views = pd.to_numeric(st.get("viewCount", 0))
        likes = pd.to_numeric(st.get("likeCount", 0))
        comments = pd.to_numeric(st.get("commentCount", 0))
        ts = pd.to_datetime(s["publishedAt"]).tz_convert(None)
        rows.append({
            "video_id": it["id"],
            "title": s["title"],
            "title_len": len(s["title"]),
            "published_at": ts,
            "publish_hour": ts.hour,
            "publish_dow": ts.dayofweek,
            "duration_seconds": dur_s,
            "views": views, "likes": likes, "comments": comments,
            "likes_per_1k_views": likes / np.maximum(views, 1) * 1000,
            "log_views": np.log1p(views),
            "peak_hour": int(ts.hour in [18,19,20,21,22])
        })
    return pd.DataFrame(rows)
