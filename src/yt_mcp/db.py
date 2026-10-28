import re
import sqlite3

from .config import CHROMA_PATH, DB_PATH

_vector_collection = None

def get_vector_collection():
    global _vector_collection
    if _vector_collection is None:
        import chromadb
        from chromadb.config import Settings
        from chromadb.utils import embedding_functions
        client = chromadb.PersistentClient(
            path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
        )
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        _vector_collection = client.get_or_create_collection(
            name="youtube_chunks",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"}
        )
    return _vector_collection

def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    with conn:
        # Tier 1: Lightweight metadata catalog of ALL watched videos
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history_catalog (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                channel TEXT,
                url TEXT,
                watched_at TEXT,
                tags TEXT DEFAULT '',
                status TEXT DEFAULT 'UNINDEXED'
            )
        """)
        # Tier 2: Full granular transcripts ONLY for user-selected videos
        conn.execute("""
            CREATE TABLE IF NOT EXISTS video_transcripts (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                channel TEXT,
                url TEXT,
                watched_at TEXT,
                tags TEXT,
                segments_json TEXT
            )
        """)
        # Tier 2: Chunk-level FTS5 keyword index
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                video_id UNINDEXED,
                title,
                channel,
                tags,
                timestamp_str UNINDEXED,
                url UNINDEXED,
                text,
                tokenize='porter unicode61'
            )
        """)
    return conn

def chunk_transcript(fetched_transcript, window_sec: int = 60, overlap_sec: int = 15) -> list[dict]:
    if window_sec <= 0 or overlap_sec < 0 or overlap_sec >= window_sec:
        raise ValueError("Require window > 0 and 0 <= overlap < window")
    snippets = sorted(list(fetched_transcript), key=lambda s: s.start)
    if not snippets:
        return []
    chunks = []
    step = window_sec - overlap_sec
    max_time = max(s.start + max(s.duration, 0.001) for s in snippets)
    current = 0
    while current < max_time:
        window = [s for s in snippets if current <= s.start < current + window_sec]
        text = " ".join(s.text.replace("\n", " ") for s in window).strip()
        if text:
            chunks.append({"start_sec": int(window[0].start), "text": text})
        if current + window_sec >= max_time:
            break
        current += step
    return chunks


def build_fts_query(query: str) -> str:
    # Quote tokens so punctuation and FTS operators in prose cannot alter syntax.
    tokens = re.findall(r"\w+", query, re.UNICODE)
    return " OR ".join('"' + token + '"' for token in tokens)
