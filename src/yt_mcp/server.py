import argparse
import json
import sys
from contextlib import closing

from mcp.server.fastmcp import FastMCP

from .db import build_fts_query, get_conn, get_vector_collection
from .sync import (
    catalog_live_youtube,
    catalog_takeout_metadata,
    index_single_video,
    interactive_pick_and_index,
    preview_video_for_fzf,
)

mcp = FastMCP("yt-transcript-mcp")

@mcp.tool()
def search_watched_videos(query: str, top_k: int = 8) -> str:
    """
    Searches curated YouTube transcripts using Hybrid RRF (SQLite FTS5 + ChromaDB).
    Returns video_id, title, channel, local_transcript_uri, clickable youtube_url with timestamp, and snippet.
    """
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    if not query.strip():
        return "[]"
    scores, registry = {}, {}
    with closing(get_conn()) as conn:
        committed = conn.execute("""SELECT COUNT(*) FROM chunks_fts f JOIN history_catalog h
            ON f.video_id=h.video_id WHERE h.status='INDEXED'""").fetchone()[0]
        if not committed:
            return "[]"
        fts_query = build_fts_query(query)
        if not fts_query:
            return "[]"
        rows = conn.execute("""SELECT chunk_id, f.video_id, f.title, f.channel,
            timestamp_str, f.url, text FROM chunks_fts f JOIN history_catalog h
            ON f.video_id=h.video_id WHERE chunks_fts MATCH ? AND h.status='INDEXED'
            ORDER BY bm25(chunks_fts,0,0,5,2,2,0,0,1) LIMIT ?""",
            (fts_query, max(20, top_k * 3))).fetchall()
        for rank, row in enumerate(rows, 1):
            cid, video_id, title, channel, stamp, url, text = row
            scores[cid] = 1 / (60 + rank)
            registry[cid] = {"video_id": video_id, "title": title, "channel": channel,
                             "timestamp": stamp, "youtube_url": url,
                             "local_transcript_uri": f"transcript://{video_id}", "snippet": text}
        collection = get_vector_collection()
        count = collection.count()
        if count:
            results = collection.query(query_texts=[query], n_results=min(count, max(20, top_k * 3)))
            for rank, cid in enumerate(results["ids"][0], 1):
                # Chroma is derived. Never expose uncommitted or obsolete vector rows.
                row = conn.execute("""SELECT f.video_id,f.title,f.channel,timestamp_str,f.url,text
                    FROM chunks_fts f JOIN history_catalog h ON h.video_id=f.video_id
                    WHERE chunk_id=? AND h.status='INDEXED'""", (cid,)).fetchone()
                if row is None:
                    continue
                video_id, title, channel, stamp, url, text = row
                scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
                registry[cid] = {"video_id": video_id, "title": title, "channel": channel,
                                 "timestamp": stamp, "youtube_url": url,
                                 "local_transcript_uri": f"transcript://{video_id}", "snippet": text}
    ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))[:top_k]
    return json.dumps([registry[cid] for cid in ranked], indent=2)

@mcp.tool()
def get_video_transcript(video_id: str, start_sec: int = 0, end_sec: int = 99999) -> str:
    """
    Fetches the full or time-sliced transcript for a video_id from the local SQLite database.
    Use start_sec and end_sec to inspect specific sections of a video.
    """
    if start_sec < 0 or end_sec < start_sec:
        raise ValueError("Require 0 <= start_sec <= end_sec")
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT title, channel, url, segments_json FROM video_transcripts WHERE video_id = ?", (video_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return f"Video ID '{video_id}' not found in local database."

    title, channel, url, segments_json = row
    segments = json.loads(segments_json)
    lines = [
        f"[{int(s['start'])//60:02d}:{int(s['start'])%60:02d} ({s['start']}s)] {s['text']}"
        for s in segments if start_sec <= s["start"] <= end_sec
    ]
    header = (
        f"Title: {title}\nChannel: {channel}\nYouTube URL: {url}\n"
        f"Local URI: transcript://{video_id}?start={start_sec}&end={end_sec}\n"
        f"{'='*60}\n"
    )
    return header + "\n".join(lines)

@mcp.tool()
def browse_unindexed_history(keyword_filter: str = "", channel_filter: str = "", limit: int = 25) -> str:
    """Lists unindexed videos from the user's watch history catalog so they can choose which ones to index."""
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT video_id, title, channel, watched_at, url
        FROM history_catalog
        WHERE status != 'INDEXED' AND title LIKE ? AND channel LIKE ?
        ORDER BY watched_at DESC LIMIT ?
    """, (f"%{keyword_filter}%", f"%{channel_filter}%", limit))
    rows = [
        {"video_id": r[0], "title": r[1], "channel": r[2], "watched_at": r[3], "url": r[4]}
        for r in cursor.fetchall()
    ]
    conn.close()
    return json.dumps(rows, indent=2)

@mcp.tool()
def add_videos_to_db(video_urls_or_ids: list[str], tags: str = "") -> str:
    """Indexes specific YouTube URLs or video_ids chosen by the user into SQLite FTS5 + ChromaDB."""
    results = [index_single_video(v, tags=tags) for v in video_urls_or_ids]
    return json.dumps(results, indent=2)

@mcp.resource("transcript://{video_id}")
def transcript_resource(video_id: str) -> str:
    return get_video_transcript(video_id)

def _main():
    parser = argparse.ArgumentParser(prog="yt-mcp", description="Curated YouTube Transcript Hybrid MCP Server")
    sub = parser.add_subparsers(dest="cmd")

    p_add = sub.add_parser("add", help="Index one or more YouTube URLs or Video IDs immediately")
    p_add.add_argument("urls", nargs="+", help="YouTube URLs or 11-char Video IDs")
    p_add.add_argument("--force", action="store_true", help="Rebuild existing transcript/indexes")
    p_add.add_argument("--tags", default="", help="Optional comma-separated tags")

    p_pick = sub.add_parser("pick", help="Interactive fzf picker with live transcript preview")
    p_pick.add_argument("--filter", default="", help="Initial keyword filter")
    p_pick.add_argument("--tags", default="", help="Tags to apply to selected videos")

    p_prev = sub.add_parser("preview", help="Internal helper called by fzf --preview")
    p_prev.add_argument("video_id")

    p_takeout = sub.add_parser("catalog-takeout", help="Load Google Takeout watch-history.json into staging catalog")
    p_takeout.add_argument("path", help="Path to watch-history.json")

    p_live = sub.add_parser("catalog-live", help="Load recent history from browser cookies into staging catalog")
    p_live.add_argument("--browser", default="chrome", help="chrome, brave, firefox, safari, edge")
    p_live.add_argument("--limit", type=int, default=100)

    p_search = sub.add_parser("search", help="Run Hybrid RRF search from CLI")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=8)

    p_get = sub.add_parser("get", help="Fetch transcript from local DB")
    p_get.add_argument("video_id")
    p_get.add_argument("--start", type=int, default=0)
    p_get.add_argument("--end", type=int, default=99999)

    sub.add_parser("doctor", help="Check runtime and local storage")
    sub.add_parser("status", help="Show catalog counts")
    args = parser.parse_args()
    if args.cmd == "add":
        failed = False
        for u in args.urls:
            result = index_single_video(u, tags=args.tags, force=args.force)
            print(json.dumps(result, indent=2))
            failed |= result["status"] not in {"INDEXED", "ALREADY_INDEXED"}
        if failed:
            raise SystemExit(1)
    elif args.cmd == "pick":
        interactive_pick_and_index(args.filter, args.tags)
    elif args.cmd == "preview":
        preview_video_for_fzf(args.video_id)
    elif args.cmd == "catalog-takeout":
        print(json.dumps(catalog_takeout_metadata(args.path)))
    elif args.cmd == "catalog-live":
        print(json.dumps(catalog_live_youtube(args.browser, args.limit)))
    elif args.cmd == "search":
        print(search_watched_videos(args.query, args.top_k))
    elif args.cmd == "get":
        print(get_video_transcript(args.video_id, args.start, args.end))
    elif args.cmd == "status":
        with closing(get_conn()) as conn:
            print(json.dumps(dict(conn.execute(
                "SELECT status,COUNT(*) FROM history_catalog GROUP BY status"
            ).fetchall()), indent=2))
    elif args.cmd == "doctor":
        import shutil
        import sqlite3

        from .config import DATA_DIR
        with closing(get_conn()) as conn:
            conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()
        print(json.dumps({"python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version,
                          "fts5": True, "fzf": shutil.which("fzf"),
                          "data_dir": str(DATA_DIR)}, indent=2))
    else:
        mcp.run(transport="stdio")

def main():
    try:
        _main()
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
