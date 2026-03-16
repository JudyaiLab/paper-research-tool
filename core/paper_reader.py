"""
Paper Reader — Shared module for reading paper content

Handles: local file reading, URL fetching, content truncation.
"""

import os
import urllib.request
from pathlib import Path

CONTENT_CHAR_LIMIT = 15000


def read_paper(source: str) -> str:
    """
    Read paper content from file path or URL.

    Args:
        source: File path or URL starting with http:// or https://

    Returns:
        Paper text content (truncated to CONTENT_CHAR_LIMIT)
    """
    if source.startswith("http://") or source.startswith("https://"):
        return _fetch_url(source)

    # Resolve to absolute path (handles ~, .., symlinks)
    try:
        path = Path(os.path.expanduser(source)).resolve(strict=False)
    except (OSError, ValueError):
        return f"[Invalid path: {source}]"

    if not path.is_file():
        return f"[File not found: {source}]"

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    if len(content) > CONTENT_CHAR_LIMIT:
        content = content[:CONTENT_CHAR_LIMIT] + f"\n\n[... Content truncated at {CONTENT_CHAR_LIMIT} characters ...]"
    return content


def _fetch_url(url: str) -> str:
    """
    Fetch text content from URL.

    Args:
        url: HTTP/HTTPS URL

    Returns:
        Text content (truncated to CONTENT_CHAR_LIMIT)
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PaperResearchTool/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode("utf-8", errors="replace")
        if len(content) > CONTENT_CHAR_LIMIT:
            content = content[:CONTENT_CHAR_LIMIT] + "\n\n[... Content truncated ...]"
        return content
    except Exception as e:
        return f"[Cannot fetch URL: {url} — {e}]"
