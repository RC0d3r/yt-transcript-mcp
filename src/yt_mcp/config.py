"""Paths only. Importing the package never creates storage."""
import os
from pathlib import Path

DATA_DIR = Path(os.getenv("YT_MCP_DATA_DIR", str(Path.home() / ".yt-memory"))).expanduser()
DB_PATH = DATA_DIR / "youtube_hybrid.db"
CHROMA_PATH = str(DATA_DIR / "chroma_db")
