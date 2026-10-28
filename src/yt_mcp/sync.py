"""Metadata staging, retryable indexing, and an fzf curation interface."""
import fcntl
import json
import re
import shlex
import shutil
import subprocess
import sys
from contextlib import closing
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi

from .db import chunk_transcript, get_conn, get_vector_collection


def extract_video_id(url_or_id: str) -> str:
    value = url_or_id.strip()
    markdown = re.fullmatch(r"\[[^\]]*\]\((https?://[^\s)]+)\)", value)
    if markdown:
        value = markdown[1]
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    parsed = urlparse(value)
    video_id = None
    if parsed.scheme in {"https", "http"} and parsed.hostname in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/")
    elif parsed.scheme in {"https", "http"} and parsed.hostname in {
        "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    }:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        else:
            match = re.fullmatch(r"/(?:shorts|embed|live)/([^/]+)/?", parsed.path)
            if match:
                video_id = match[1]
    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Expected a YouTube URL or an 11-character video ID")
    return video_id


def _run_ytdlp(arguments: list[str], timeout: int = 120):
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--ignore-config", *arguments], capture_output=True,
        text=True, timeout=timeout, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()[-1500:]}")
    return result


def fetch_video_metadata_ytdlp(video_id: str) -> dict:
    video_id = extract_video_id(video_id)
    result = _run_ytdlp([
        "--no-warnings", "--dump-single-json", "--skip-download", "--no-playlist",
        f"https://www.youtube.com/watch?v={video_id}",
    ])
    data = json.loads(result.stdout)
    return {"title": data.get("title") or f"Video {video_id}",
            "channel": data.get("uploader") or data.get("channel") or "Unknown"}


def index_single_video(url_or_id: str, tags: str = "", force: bool = False) -> dict:
    try:
        video_id = extract_video_id(url_or_id)
    except ValueError as exc:
        return {"status": "INVALID_URL", "input": url_or_id, "error": str(exc)}
    from . import db
    db.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(db.DB_PATH.parent / ".index.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"video_id": video_id, "status": "INDEX_BUSY",
                    "error": "Another indexing operation is active; retry when it finishes"}
        try:
            return _index_single_video(video_id, tags, force)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _index_single_video(url_or_id: str, tags: str = "", force: bool = False) -> dict:
    try:
        video_id = extract_video_id(url_or_id)
    except ValueError as exc:
        return {"status": "INVALID_URL", "input": url_or_id, "error": str(exc)}
    url = f"https://www.youtube.com/watch?v={video_id}"
    with closing(get_conn()) as conn:
        row = conn.execute(
            "SELECT title, channel, watched_at, status, tags FROM history_catalog WHERE video_id=?",
            (video_id,),
        ).fetchone()
        if row and row[3] == "INDEXED" and not force:
            return {"video_id": video_id, "status": "ALREADY_INDEXED", "title": row[0]}
        if row:
            title, channel, watched_at = row[:3]
            tags = tags or row[4]
        if not row or (row[3] == "FETCH_ERROR" and row[0] == f"Video {video_id}" and row[1] == "Unknown"):
            title, channel, watched_at = f"Video {video_id}", "Unknown", "manual_add"
            try:
                meta = fetch_video_metadata_ytdlp(video_id)
                title, channel = meta["title"], meta["channel"]
            except Exception as exc:
                return _record_failure(conn, video_id, title, channel, url, watched_at,
                                       tags, "FETCH_ERROR", exc)
        try:
            fetched = list(YouTubeTranscriptApi().fetch(video_id, languages=["en", "en-US", "en-GB"]))
        except (NoTranscriptFound, TranscriptsDisabled) as exc:
            return _record_failure(conn, video_id, title, channel, url, watched_at,
                                   tags, "NO_TRANSCRIPT", exc)
        except Exception as exc:
            return _record_failure(conn, video_id, title, channel, url, watched_at,
                                   tags, "FETCH_ERROR", exc)
        chunks = chunk_transcript(fetched)
        if not chunks:
            return _record_failure(conn, video_id, title, channel, url, watched_at,
                                   tags, "NO_TRANSCRIPT", ValueError("Empty transcript"))
        segments = [{"start": s.start, "duration": s.duration,
                     "text": s.text.replace("\n", " ")} for s in fetched]
        with conn:
            conn.execute("""INSERT INTO history_catalog
                (video_id,title,channel,url,watched_at,tags,status) VALUES (?,?,?,?,?,?,'INDEXING')
                ON CONFLICT(video_id) DO UPDATE SET status='INDEXING',title=excluded.title,channel=excluded.channel""",
                (video_id, title, channel, url, watched_at, tags))
        try:
            ids, docs, metas, rows = [], [], [], []
            for index, chunk in enumerate(chunks):
                start = chunk["start_sec"]
                stamp = f"{start // 60:02d}:{start % 60:02d}"
                link = f"{url}&t={start}s"
                chunk_id = f"{video_id}_{start}_{index}"
                ids.append(chunk_id)
                docs.append(f"{title} ({channel}) [{tags}]: {chunk['text']}")
                metas.append({"video_id": video_id, "title": title, "channel": channel,
                              "tags": tags, "timestamp": stamp, "url": link,
                              "raw_text": chunk["text"]})
                rows.append((chunk_id, video_id, title, channel, tags, stamp, link, chunk["text"]))
            collection = get_vector_collection()
            collection.delete(where={"video_id": video_id})
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
            with conn:
                conn.execute("DELETE FROM chunks_fts WHERE video_id=?", (video_id,))
                conn.executemany("INSERT INTO chunks_fts VALUES (?,?,?,?,?,?,?,?)", rows)
                conn.execute("INSERT OR REPLACE INTO video_transcripts VALUES (?,?,?,?,?,?,?)",
                             (video_id, title, channel, url, watched_at, tags, json.dumps(segments)))
                conn.execute("UPDATE history_catalog SET status='INDEXED',tags=? WHERE video_id=?",
                             (tags, video_id))
            return {"video_id": video_id, "title": title, "status": "INDEXED", "chunks": len(chunks)}
        except Exception as exc:
            return _record_failure(conn, video_id, title, channel, url, watched_at,
                                   tags, "INDEX_ERROR", exc)


def _record_failure(conn, video_id, title, channel, url, watched_at, tags, status, exc):
    with conn:
        conn.execute("""INSERT INTO history_catalog VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(video_id) DO UPDATE SET status=CASE WHEN history_catalog.status='INDEXED'
            THEN history_catalog.status ELSE excluded.status END""",
            (video_id, title, channel, url, watched_at, tags, status))
    return {"video_id": video_id, "title": title, "status": status, "error": str(exc)}


def _stage(rows: list[tuple]) -> dict:
    with closing(get_conn()) as conn, conn:
        before = conn.total_changes
        conn.executemany("""INSERT INTO history_catalog
            (video_id,title,channel,url,watched_at) VALUES (?,?,?,?,?)
            ON CONFLICT(video_id) DO NOTHING""", rows)
        return {"imported": conn.total_changes - before, "valid_entries": len(rows)}


def catalog_takeout_metadata(json_path: str) -> dict:
    with open(json_path, encoding="utf-8") as file:
        history = json.load(file)
    if not isinstance(history, list):
        raise ValueError("Expected a JSON array from Google Takeout")
    rows = []
    for entry in history:
        if not isinstance(entry, dict) or entry.get("header") == "YouTube Music":
            continue
        try:
            video_id = extract_video_id(entry.get("titleUrl", ""))
        except ValueError:
            continue
        subs = entry.get("subtitles") or [{}]
        rows.append((video_id, entry.get("title", "").removeprefix("Watched "),
                     subs[0].get("name", "Unknown"),
                     f"https://www.youtube.com/watch?v={video_id}", entry.get("time", "")))
    return _stage(rows)


def catalog_live_youtube(browser: str = "chrome", limit: int = 100) -> dict:
    if browser not in {"chrome", "brave", "firefox", "safari", "edge", "chromium", "vivaldi"}:
        raise ValueError("Unsupported browser")
    if not 1 <= limit <= 10000:
        raise ValueError("limit must be between 1 and 10000")
    result = _run_ytdlp([
        "--cookies-from-browser", browser, "--flat-playlist", "--dump-json",
        "--playlist-end", str(limit), "https://www.youtube.com/feed/history",
    ], timeout=300)
    rows = []
    for line in result.stdout.splitlines():
        data = json.loads(line)
        try:
            video_id = extract_video_id(data.get("id", ""))
        except ValueError:
            continue
        rows.append((video_id, data.get("title") or "Unknown",
                     data.get("uploader") or data.get("channel") or "Unknown",
                     f"https://www.youtube.com/watch?v={video_id}", "recent"))
    return _stage(rows)


def _display(text):
    # Strip terminal control characters from externally supplied metadata/captions.
    return "".join(c if c.isprintable() else " " for c in str(text))


def preview_video_for_fzf(video_id: str):
    video_id = extract_video_id(video_id)
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT title,channel,status,url FROM history_catalog WHERE video_id=?",
                           (video_id,)).fetchone()
        stored = conn.execute("SELECT segments_json FROM video_transcripts WHERE video_id=?",
                              (video_id,)).fetchone()
    if not row:
        print("No catalog metadata for this video")
        return
    print("\n".join(f"{label}: {_display(value)}" for label, value in
                    zip(("Title", "Channel", "Status", "URL"), row)))
    try:
        segments = json.loads(stored[0]) if stored else [
            {"start": s.start, "text": s.text}
            for s in YouTubeTranscriptApi().fetch(video_id, languages=["en", "en-US", "en-GB"])
        ]
        print("\nLocal transcript" if stored else "\nLive transcript preview")
        for seg in segments[:25]:
            start = int(seg["start"])
            print(f"[{start // 60:02d}:{start % 60:02d}] {_display(seg['text'])}")
    except Exception as exc:
        print(f"Preview unavailable: {_display(exc)}")


def interactive_pick_and_index(keyword: str = "", tags: str = ""):
    if not shutil.which("fzf"):
        raise RuntimeError("fzf is required: brew install fzf")
    with closing(get_conn()) as conn:
        rows = conn.execute("""SELECT video_id,watched_at,channel,title FROM history_catalog
            WHERE status != 'INDEXED' AND (title LIKE ? OR channel LIKE ?)
            ORDER BY watched_at DESC""", (f"%{keyword}%", f"%{keyword}%")).fetchall()
    if not rows:
        print("No pending videos. Import history with catalog-takeout or catalog-live.")
        return
    lines = ["\t".join(_display(v) for v in row) for row in rows]
    # Only the validated, fixed-format first field enters preview/open shell commands.
    launcher = f"{shlex.quote(sys.executable)} -m yt_mcp.server"
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    result = subprocess.run([
        "fzf", "--multi", "--height=90%", "--layout=reverse", "--border",
        "--delimiter=\t", "--with-nth=2..", "--prompt=Filter history > ",
        "--header=Tab: select | Enter: index | Ctrl-O: open | Ctrl-/: preview",
        f"--preview={launcher} preview {{1}}", "--preview-window=right:45%:wrap",
        "--bind=ctrl-/:toggle-preview",
        f"--bind=ctrl-o:execute-silent({opener} 'https://www.youtube.com/watch?v='{{1}})",
    ], input="\n".join(lines), text=True, stdout=subprocess.PIPE, check=False)
    if result.returncode in {1, 130}:
        return
    if result.returncode:
        raise RuntimeError(f"fzf exited with {result.returncode}")
    failed = False
    for line in result.stdout.splitlines():
        indexed = index_single_video(line.split("\t")[0], tags=tags)
        print(json.dumps(indexed))
        failed |= indexed["status"] not in {"INDEXED", "ALREADY_INDEXED"}
    if failed:
        raise RuntimeError("Some selected videos failed to index; see statuses above")
