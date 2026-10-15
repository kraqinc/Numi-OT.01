"""Local RAG: learn new facts by dropping files; retrieve into the prompt."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LEARN_DIR = _ROOT / "knowledge" / "learn"
_DEFAULT_DIRS = [
    _LEARN_DIR,
    _ROOT / "knowledge" / "sources",
]

_cache_chunks: list[tuple[str, str]] = []
_cache_mtime: float = 0.0


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ0-9]{2,}", text.lower()))


def _dirs() -> list[Path]:
    extra = os.getenv("NUMI_KNOWLEDGE_DIRS", "")
    dirs = list(_DEFAULT_DIRS)
    for part in extra.split(":"):
        part = part.strip()
        if part:
            dirs.append(Path(part))
    return dirs


def _scan_mtime(dirs: list[Path]) -> float:
    latest = 0.0
    for d in dirs:
        if not d.exists():
            continue
        for path in d.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".md", ".txt", ".jsonl"}:
                try:
                    latest = max(latest, path.stat().st_mtime)
                except OSError:
                    pass
    return latest


def load_chunks(dirs: list[Path] | None = None, force: bool = False) -> list[tuple[str, str]]:
    global _cache_chunks, _cache_mtime
    use_dirs = dirs or _dirs()
    mtime = _scan_mtime(use_dirs)
    if not force and _cache_chunks and mtime <= _cache_mtime:
        return _cache_chunks

    chunks: list[tuple[str, str]] = []
    for d in use_dirs:
        if not d.exists():
            continue
        for path in sorted(d.rglob("*")):
            if path.suffix.lower() not in {".md", ".txt", ".jsonl"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if path.suffix == ".jsonl":
                for i, line in enumerate(text.splitlines()):
                    line = line.strip()
                    if line:
                        chunks.append((f"{path.name}:{i}", line[:3000]))
            else:
                for i, para in enumerate(re.split(r"\n\s*\n", text)):
                    para = para.strip()
                    if len(para) > 20:
                        chunks.append((f"{path.name}:{i}", para[:3000]))
    _cache_chunks = chunks
    _cache_mtime = mtime or time.time()
    return chunks


def retrieve(query: str, k: int = 5) -> str:
    q = _tokenize(query)
    if not q:
        return ""
    scored: list[tuple[int, str, str]] = []
    for name, chunk in load_chunks():
        overlap = len(q & _tokenize(chunk))
        if overlap:
            scored.append((overlap, name, chunk))
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[:k]
    if not top:
        return ""
    return "\n\n".join(f"[{name}] {chunk}" for _, name, chunk in top)


def inject_context(user_text: str, k: int = 5) -> str:
    if os.getenv("NUMI_RAG", "1") == "0":
        return user_text
    ctx = retrieve(user_text, k=k)
    if not ctx:
        return user_text
    return (
        "Use facts from CONTEXT when relevant. If context is missing or weak, say so; "
        "do not invent.
"
        f"### CONTEXT\n{ctx}\n\n### USER\n{user_text}"
    )


def invalidate_cache() -> None:
    global _cache_chunks, _cache_mtime
    _cache_chunks = []
    _cache_mtime = 0.0
