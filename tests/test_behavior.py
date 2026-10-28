"""Independent behavior acceptance. No network or real model downloads."""
import importlib
import json
from types import SimpleNamespace

import pytest

VIDEO = "dQw4w9WgXcQ"
OTHER = "jNQXAC9IVRw"


class Vectors:
    def __init__(self):
        self.rows = {}
        self.fail = False
        self.queries = []

    def upsert(self, ids, documents, metadatas):
        if self.fail:
            raise RuntimeError("injected vector failure")
        self.rows.update({key: (doc, meta) for key, doc, meta in zip(ids, documents, metadatas)})

    def delete(self, ids=None, where=None):
        for key in list(self.rows):
            if (ids is not None and key in ids) or (where and self.rows[key][1].get("video_id") == where.get("video_id")):
                self.rows.pop(key)

    def count(self):
        return len(self.rows)

    def query(self, query_texts, n_results, **kwargs):
        self.queries.append((query_texts, n_results))
        assert 0 < n_results <= self.count(), "vector query must be bounded by collection count"
        keys = list(self.rows)[:n_results]
        return {"ids": [keys], "metadatas": [[self.rows[key][1] for key in keys]]}


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("YT_MCP_DATA_DIR", str(tmp_path / "data"))
    db = importlib.import_module("yt_mcp.db")
    sync = importlib.import_module("yt_mcp.sync")
    server = importlib.import_module("yt_mcp.server")
    (tmp_path / "data").mkdir(exist_ok=True)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "data" / "test.db")
    monkeypatch.setattr(db, "CHROMA_PATH", str(tmp_path / "data" / "chroma"))
    vectors = Vectors()
    monkeypatch.setattr(db, "get_vector_collection", lambda: vectors)
    monkeypatch.setattr(sync, "get_vector_collection", lambda: vectors)
    monkeypatch.setattr(server, "get_vector_collection", lambda: vectors)
    snippets = [
        SimpleNamespace(start=0.0, duration=4.0, text="A concise systems lesson."),
        SimpleNamespace(start=46.0, duration=5.0, text="Distributed queues provide durable event processing."),
        SimpleNamespace(start=100.0, duration=7.0, text="Reciprocal rank fusion merges independent rankings."),
    ]
    monkeypatch.setattr(sync, "YouTubeTranscriptApi", lambda: SimpleNamespace(fetch=lambda *a, **k: snippets))
    monkeypatch.setattr(sync, "fetch_video_metadata_ytdlp", lambda *a, **k: {"title": "System design lesson", "channel": "Test channel"})
    return SimpleNamespace(db=db, sync=sync, server=server, vectors=vectors, snippets=snippets, tmp=tmp_path)


@pytest.mark.parametrize("value", [
    VIDEO,
    f"https://www.youtube.com/watch?v={VIDEO}&t=8s",
    f"https://youtu.be/{VIDEO}?t=8",
    f"https://m.youtube.com/shorts/{VIDEO}",
    f"https://www.youtube.com/embed/{VIDEO}",
    f"https://www.youtube.com/live/{VIDEO}",
    f"[https://www.youtube.com/watch?v={VIDEO}](https://www.youtube.com/watch?v={VIDEO})",
])
def test_youtube_input_forms(app, value):
    assert app.sync.extract_video_id(value) == VIDEO


@pytest.mark.parametrize("value", [
    "", "garbage", "hello world", "https://example.com/watch?v=" + VIDEO,
    "https://youtube.com.evil.example/watch?v=" + VIDEO,
    "https://youtube.com/watch", "https://youtu.be/not-valid", "--exec=evil",
    "https://www.youtube.com/watch?v=../../evil", "abcdefghij;",
])
def test_untrusted_ids_rejected(app, value):
    with pytest.raises(ValueError):
        app.sync.extract_video_id(value)


def test_short_speech_is_not_dropped(app):
    chunks = app.db.chunk_transcript([SimpleNamespace(start=0, duration=2, text="Short lesson.")])
    assert chunks and chunks[0]["text"] == "Short lesson."
    assert chunks[0]["start_sec"] == 0


@pytest.mark.parametrize("window,overlap", [(0, 0), (-1, 0), (60, 60), (60, 61), (60, -1)])
def test_bad_chunk_window_rejected(app, window, overlap):
    with pytest.raises(ValueError):
        app.db.chunk_transcript(app.snippets, window_sec=window, overlap_sec=overlap)


def test_chunk_overlap_and_order(app):
    chunks = app.db.chunk_transcript(app.snippets)
    assert chunks[0]["start_sec"] == 0
    assert any("durable event processing" in chunk["text"] for chunk in chunks)
    assert any("independent rankings" in chunk["text"] for chunk in chunks)
    assert [chunk["start_sec"] for chunk in chunks] == sorted(chunk["start_sec"] for chunk in chunks)


def test_metadata_catalog_is_lazy_idempotent_and_preserves_selection(app, monkeypatch):
    def forbidden():
        raise AssertionError("metadata import must never initialize embeddings")
    monkeypatch.setattr(app.sync, "get_vector_collection", forbidden)
    entries = [
        {"header": "YouTube", "titleUrl": f"https://www.youtube.com/watch?v={VIDEO}", "title": "Watched Queue systems", "time": "2026-10-27T10:11:12Z", "subtitles": [{"name": "Engineer"}]},
        {"header": "YouTube", "titleUrl": f"https://www.youtube.com/watch?v={VIDEO}", "title": "Watched Queue systems", "time": "2026-10-27T10:11:12Z"},
        {"header": "YouTube", "titleUrl": "https://example.com/fake", "title": "Bad"},
        {"header": "YouTube Music", "titleUrl": f"https://www.youtube.com/watch?v={OTHER}", "title": "Music"},
    ]
    path = app.tmp / "takeout.json"
    path.write_text(json.dumps(entries))
    result = app.sync.catalog_takeout_metadata(str(path))
    assert result["imported"] == 1
    with app.db.get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM history_catalog").fetchone()[0] == 1
        conn.execute("UPDATE history_catalog SET status='INDEXED',tags='saved'")
        assert conn.execute("SELECT COUNT(*) FROM video_transcripts").fetchone()[0] == 0
    result = app.sync.catalog_takeout_metadata(str(path))
    assert result["imported"] == 0
    with app.db.get_conn() as conn:
        assert conn.execute("SELECT status,tags FROM history_catalog").fetchone() == ("INDEXED", "saved")


def test_index_populates_both_stores_once_with_timestamp_links(app):
    result = app.sync.index_single_video(VIDEO, tags="systems")
    assert result["status"] == "INDEXED"
    assert result["chunks"] > 0
    with app.db.get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        assert count == len(app.vectors.rows) == result["chunks"]
        assert conn.execute("SELECT status,tags FROM history_catalog").fetchone() == ("INDEXED", "systems")
        assert conn.execute("SELECT COUNT(*) FROM video_transcripts").fetchone()[0] == 1
    again = app.sync.index_single_video(VIDEO)
    assert again["status"] == "ALREADY_INDEXED"
    with app.db.get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == count
    assert all("&t=" in meta["url"] for _, meta in app.vectors.rows.values())


def test_failed_vector_write_is_retryable_and_not_success(app):
    app.vectors.fail = True
    failed = app.sync.index_single_video(VIDEO)
    assert failed["status"] == "INDEX_ERROR"
    with app.db.get_conn() as conn:
        row = conn.execute("SELECT status FROM history_catalog WHERE video_id=?", (VIDEO,)).fetchone()
        assert row and row[0] != "INDEXED"
    app.vectors.fail = False
    recovered = app.sync.index_single_video(VIDEO)
    assert recovered["status"] == "INDEXED"
    with app.db.get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == recovered["chunks"]


def test_empty_database_search_is_empty_without_vector_query(app):
    assert json.loads(app.server.search_watched_videos("distributed systems")) == []
    assert app.vectors.queries == []


@pytest.mark.parametrize("query", ["queues", '" OR * NOT ()', "a", "", "???", "C++ FTS5 λ"])
def test_arbitrary_search_input_is_safe(app, query):
    app.sync.index_single_video(VIDEO)
    assert isinstance(json.loads(app.server.search_watched_videos(query)), list)


@pytest.mark.parametrize("top_k", [0, -1, 101])
def test_search_limits_validated(app, top_k):
    with pytest.raises(ValueError):
        app.server.search_watched_videos("queues", top_k=top_k)


def test_search_returns_grounded_results_and_transcript_slice(app):
    app.sync.index_single_video(VIDEO, tags="systems")
    results = json.loads(app.server.search_watched_videos("durable event processing", top_k=2))
    assert 0 < len(results) <= 2
    for result in results:
        assert result["video_id"] == VIDEO
        assert result["local_transcript_uri"] == f"transcript://{VIDEO}"
        assert "&t=" in result["youtube_url"]
        assert result["snippet"]
    text = app.server.get_video_transcript(VIDEO, start_sec=40, end_sec=60)
    assert "Distributed queues" in text
    assert "A concise" not in text
    assert "Reciprocal rank" not in text


@pytest.mark.parametrize("start,end", [(-1, 10), (10, 9)])
def test_transcript_bad_bounds_rejected(app, start, end):
    with pytest.raises(ValueError):
        app.server.get_video_transcript(VIDEO, start_sec=start, end_sec=end)


def test_catalog_subprocess_failure_is_visible(app, monkeypatch):
    monkeypatch.setattr(app.sync.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="browser cookie failure"))
    with pytest.raises((RuntimeError, OSError)):
        app.sync.catalog_live_youtube("chrome", 10)


def test_invalid_add_never_calls_downloader(app, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid URLs must not trigger metadata or transcript downloads")
    monkeypatch.setattr(app.sync, "fetch_video_metadata_ytdlp", forbidden)
    result = app.sync.index_single_video("https://example.com/this-is-not-youtube")
    assert result["status"] == "INVALID_URL"


# Independent regressions added after review, 2026-10-28.
def test_metadata_fetch_failure_is_recorded_without_caption_or_index_writes(app, monkeypatch):
    def broken_metadata(*args, **kwargs):
        raise RuntimeError("injected metadata service failure")

    def forbidden_captions():
        raise AssertionError("metadata failure must stop before caption download")

    monkeypatch.setattr(app.sync, "fetch_video_metadata_ytdlp", broken_metadata)
    monkeypatch.setattr(app.sync, "YouTubeTranscriptApi", forbidden_captions)
    result = app.sync.index_single_video(VIDEO)
    assert result["status"] == "FETCH_ERROR"
    assert "metadata service failure" in result["error"]
    with app.db.get_conn() as conn:
        assert conn.execute("SELECT status FROM history_catalog WHERE video_id=?", (VIDEO,)).fetchone()[0] == "FETCH_ERROR"
        assert conn.execute("SELECT COUNT(*) FROM video_transcripts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 0
    assert app.vectors.count() == 0


def test_metadata_downloader_ignores_global_ytdlp_config(monkeypatch):
    sync = importlib.import_module("yt_mcp.sync")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"title": "Safe metadata", "uploader": "Safe channel"}), stderr="")

    monkeypatch.setattr(sync.subprocess, "run", run)
    result = sync.fetch_video_metadata_ytdlp(VIDEO)
    assert result == {"title": "Safe metadata", "channel": "Safe channel"}
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert "--ignore-config" in argv
    assert "--skip-download" in argv
    assert "--no-playlist" in argv
    assert kwargs.get("shell", False) is False
    assert kwargs["timeout"] > 0
    assert argv[-1] == f"https://www.youtube.com/watch?v={VIDEO}"


def test_history_downloader_ignores_global_ytdlp_config(app, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"id": VIDEO, "title": "Safe history", "uploader": "Safe channel"}) + "\n", stderr="")

    monkeypatch.setattr(app.sync.subprocess, "run", run)
    result = app.sync.catalog_live_youtube("chrome", 10)
    assert result["imported"] == 1
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert "--ignore-config" in argv
    assert "--cookies-from-browser" in argv
    assert "--flat-playlist" in argv
    assert kwargs.get("shell", False) is False
    assert kwargs["timeout"] > 0


def test_data_dir_lock_rejects_concurrent_add_without_blocking(app, monkeypatch):
    import fcntl

    lock_file = app.db.DB_PATH.parent / ".index.lock"
    real_flock = fcntl.flock
    calls = []

    def checked_flock(fd, operation):
        calls.append(operation)
        if operation & fcntl.LOCK_EX:
            assert operation & fcntl.LOCK_NB, "index lock acquisition must be nonblocking"
        return real_flock(fd, operation)

    with lock_file.open("a") as held:
        real_flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(fcntl, "flock", checked_flock)
        busy = app.sync.index_single_video(VIDEO)
        assert busy["status"] == "INDEX_BUSY"
        assert app.vectors.count() == 0
        with app.db.get_conn() as conn:
            assert conn.execute("SELECT COUNT(*) FROM history_catalog").fetchone()[0] == 0
        real_flock(held, fcntl.LOCK_UN)
    assert calls and calls[0] & fcntl.LOCK_NB
    assert app.sync.index_single_video(VIDEO)["status"] == "INDEXED"


def test_failed_forced_refresh_preserves_existing_healthy_selection(app, monkeypatch):
    assert app.sync.index_single_video(VIDEO, tags="keep-me")["status"] == "INDEXED"
    with app.db.get_conn() as conn:
        original_transcript = conn.execute("SELECT segments_json FROM video_transcripts WHERE video_id=?", (VIDEO,)).fetchone()[0]
        original_chunks = conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE video_id=?", (VIDEO,)).fetchone()[0]
    original_vectors = dict(app.vectors.rows)

    def broken_fetch(*args, **kwargs):
        raise RuntimeError("injected forced refresh fetch failure")

    monkeypatch.setattr(app.sync, "YouTubeTranscriptApi", lambda: SimpleNamespace(fetch=broken_fetch))
    result = app.sync.index_single_video(VIDEO, force=True)
    assert result["status"] == "FETCH_ERROR"
    with app.db.get_conn() as conn:
        assert conn.execute("SELECT status,tags FROM history_catalog WHERE video_id=?", (VIDEO,)).fetchone() == ("INDEXED", "keep-me")
        assert conn.execute("SELECT segments_json FROM video_transcripts WHERE video_id=?", (VIDEO,)).fetchone()[0] == original_transcript
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE video_id=?", (VIDEO,)).fetchone()[0] == original_chunks
    assert app.vectors.rows == original_vectors
    assert json.loads(app.server.search_watched_videos("durable event processing"))


def test_retry_after_metadata_failure_refetches_real_title_and_channel(app, monkeypatch):
    attempts = []

    def metadata(video_id):
        attempts.append(video_id)
        if len(attempts) == 1:
            raise RuntimeError("temporary metadata failure")
        return {"title": "Actual recovered title", "channel": "Actual recovered channel"}

    monkeypatch.setattr(app.sync, "fetch_video_metadata_ytdlp", metadata)
    assert app.sync.index_single_video(VIDEO)["status"] == "FETCH_ERROR"
    recovered = app.sync.index_single_video(VIDEO)
    assert recovered["status"] == "INDEXED"
    assert attempts == [VIDEO, VIDEO]
    assert recovered["title"] == "Actual recovered title"
    with app.db.get_conn() as conn:
        assert conn.execute("SELECT title,channel,status FROM history_catalog WHERE video_id=?", (VIDEO,)).fetchone() == ("Actual recovered title", "Actual recovered channel", "INDEXED")
        assert conn.execute("SELECT title,channel FROM video_transcripts WHERE video_id=?", (VIDEO,)).fetchone() == ("Actual recovered title", "Actual recovered channel")
    assert all(meta["title"] == "Actual recovered title" and meta["channel"] == "Actual recovered channel" for _, meta in app.vectors.rows.values())
