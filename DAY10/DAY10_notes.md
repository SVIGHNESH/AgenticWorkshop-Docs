# Day 10 — Text Chunking for LLM Pipelines

## Theory

### What is Text Chunking?
Text chunking is the process of splitting a large document into smaller, manageable pieces before passing them to an LLM or embedding model. Most models have a fixed context window, so long documents must be broken into chunks that fit within that limit.

### Fixed-Size Chunking
Fixed-size chunking splits text every N characters regardless of sentence or paragraph boundaries. It is simple, predictable, and fast — making it a reliable baseline strategy. The trade-off is that it can cut mid-sentence, so the beginning and end of each chunk may lack full context.

### Overlap
Overlap addresses the mid-sentence cut problem by repeating the last K characters of one chunk at the start of the next. This ensures that context around chunk boundaries is preserved in at least one of the two adjacent chunks. With size=500 and overlap=50, each chunk advances by 450 characters — so consecutive chunks share a 50-character window.

### Chunk Metadata
Each chunk carries an `index`, the raw `text`, and a `char_count`. Downstream systems (vector stores, LLMs) use the index to reconstruct order and the char_count for token estimation.

### CLI Design with argparse
The tool exposes chunking parameters directly as CLI flags (`--size`, `--overlap`, `--format`). Input can come from a file (`--file`), an inline string (`--text`), or stdin — making it composable with shell pipelines.

### Output Formats
Two output formats are supported:
- **JSON** — a structured array of chunk objects, suitable for piping into other programs or storing in a vector database.
- **Plain text** — chunks separated by `---`, suitable for human review or simple downstream processing.

---

## Code Examples

### Fixed-size sliding window chunking

> What this demonstrates: core chunking logic — advance by `size - overlap` each iteration

```python
# From: Code/DAY10/ChunkingAgent/src/chunking_agent/chunker.py
def chunk(text: str, size: int = 500, overlap: int = 50) -> list[dict]:
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap

    return [
        {"index": i, "text": t, "char_count": len(t)}
        for i, t in enumerate(chunks)
    ]
```

---

### Accepting input from file, inline text, or stdin

> What this demonstrates: flexible input sourcing — a mutually exclusive argparse group with a stdin fallback

```python
# From: Code/DAY10/ChunkingAgent/src/chunking_agent/cli.py
source = parser.add_mutually_exclusive_group()
source.add_argument("--file", metavar="PATH", help="Read input from file")
source.add_argument("--text", metavar="TEXT", help="Inline input text")

if args.file:
    with open(args.file) as f:
        text = f.read()
elif args.text:
    text = args.text
else:
    if sys.stdin.isatty():
        parser.error("Provide --file, --text, or pipe input via stdin.")
    text = sys.stdin.read()
```

---

### Dual output formatters

> What this demonstrates: separating formatting logic from chunking logic — JSON for machines, text for humans

```python
# From: Code/DAY10/ChunkingAgent/src/chunking_agent/output.py
def format_json(chunks: list[dict]) -> str:
    return json.dumps(chunks, indent=2)

def format_text(chunks: list[dict]) -> str:
    return "\n---\n".join(c["text"] for c in chunks)
```

---

### Verifying overlap count in tests

> What this demonstrates: deriving expected chunk count from the sliding-window formula

```python
# From: Code/DAY10/ChunkingAgent/tests/test_chunker.py
def test_overlap():
    text = "A" * 1000
    result = chunk(text, size=500, overlap=50)
    # Each step advances by size - overlap = 450
    assert len(result) == 3  # 0..499, 450..949, 900..999

def test_last_chunk_smaller():
    text = "X" * 600
    result = chunk(text, size=500, overlap=50)
    assert result[-1]["char_count"] == 150  # 600 - (500 - 50) = 150
```

---

### Dispatching format at the CLI level

> What this demonstrates: routing output through the correct formatter based on the `--format` flag

```python
# From: Code/DAY10/ChunkingAgent/src/chunking_agent/cli.py
chunks = chunk(text, size=args.size, overlap=args.overlap)

if args.format == "json":
    print(format_json(chunks))
else:
    print(format_text(chunks))
```

---

## Project Architecture

```
DAY10/
└── ChunkingAgent/
    ├── pyproject.toml                        ← uv project config, CLI entry point registration
    ├── src/
    │   └── chunking_agent/
    │       ├── __init__.py                   ← empty package marker
    │       ├── chunker.py                    ← fixed-size sliding window chunking logic
    │       ├── cli.py                        ← argparse CLI — input sourcing + flag wiring
    │       └── output.py                     ← JSON and plain-text formatters
    └── tests/
        └── test_chunker.py                   ← unit tests for chunk size, overlap, metadata, edge cases
```
