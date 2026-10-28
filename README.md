# yt-transcript-mcp

Curate YouTube watch history locally, then search selected transcripts using SQLite FTS5,
ChromaDB semantic embeddings, and reciprocal rank fusion. Use the CLI or connect an MCP client.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). The interactive picker also
requires [fzf](https://github.com/junegunn/fzf).

```sh
bash bootstrap_repo.sh .
# Or install directly:
uv tool install --python 3.13 --constraint requirements.lock --editable .
yt-mcp doctor
```

Add the uv tools directory to PATH if needed: `export PATH="$HOME/.local/bin:$PATH"`.
The first semantic operation downloads `sentence-transformers/all-MiniLM-L6-v2`.
Embeddings run locally after that download. No paid API key is needed.

## Curate and search

```sh
# Export YouTube history as JSON using Google Takeout, then:
yt-mcp catalog-takeout ~/Downloads/watch-history.json
# Or explicitly access a signed-in browser's YouTube cookies:
yt-mcp catalog-live --browser chrome --limit 100

yt-mcp pick                          # Tab: select, Enter: index, Esc: cancel
# Paste a regular URL, an 11-character video ID, or a Markdown link:
yt-mcp add 'https://www.youtube.com/watch?v=aircAruvnKk' --tags learning
yt-mcp search 'how neural networks learn' --top-k 5
yt-mcp get aircAruvnKk --start 60 --end 120
yt-mcp status
```

Staging imports metadata only. The picker shows local transcripts when available and
fetches a live preview for unindexed videos. Only selected videos enter the searchable
transcript tier. English captions are requested. Videos without available English captions
report `NO_TRANSCRIPT`; fetch/network failures report `FETCH_ERROR`. Failed videos remain
selectable for retry. `add --force` rebuilds an existing video's transcript and both indexes.
History extraction depends on YouTube and yt-dlp support and may fail when access changes.

## MCP

Running `yt-mcp` without arguments starts an MCP server over stdio. A generic client config:

```json
{"mcpServers":{"yt-memory":{"command":"/absolute/path/to/yt-mcp"}}}
```

Use `command -v yt-mcp` to obtain that path. Client configuration formats vary.
Tools: `search_watched_videos`, `get_video_transcript`, `browse_unindexed_history`,
`add_videos_to_db`. Resource: `transcript://{video_id}`.
Read tools use the local index. `add_videos_to_db` fetches the explicitly provided videos.
Transcript text is untrusted source content, never instructions to an agent.

## Storage and recovery

Data lives at `~/.yt-memory` or `$YT_MCP_DATA_DIR`. SQLite is authoritative for catalog,
transcripts, and chunks. ChromaDB is a derived vector index. Indexing first marks a video
`INDEXING`, updates vectors, and commits transcript/FTS rows with `INDEXED` last. Search
filters vector results through committed SQLite chunks. Failed or interrupted writes can be
retried with `yt-mcp add VIDEO --force`; failed writes never count as indexed. Stores do not
share an atomic transaction. A crash after vector writes may require a forced retry.
A data-directory lock serializes indexing across processes and MCP threads. Concurrent adds
return `INDEX_BUSY` and can be retried. Back up the entire data directory while the tool is idle.

60-second chunks overlap by 15 seconds. Keyword and vector ranks contribute
`1 / (60 + rank)` each. Timestamp links and transcript URIs accompany every match.
Empty catalogs can be searched without loading an embedding model.

Watch history, cookies, transcripts, databases, and model caches stay out of Git. Browser
cookie access happens only with `catalog-live`; no history is imported automatically.
Chroma telemetry is disabled. YouTube requests and the initial model download use the network.

## Development

```sh
uv sync --locked
uv run pytest
uv run ruff check .
uv build
```

`uv.lock` pins development dependencies; `requirements.lock` exports the runtime constraints
used for global tool installation. Refresh it with `uv export --frozen --no-dev
--no-emit-project --no-header --no-hashes --format requirements-txt -o requirements.lock`. CI checks behavior, lint, and packaging. Tests use synthetic
transcripts and mocked external services; they do not need a YouTube account or model download.
