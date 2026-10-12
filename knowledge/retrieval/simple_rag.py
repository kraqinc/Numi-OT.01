"""Minimal local RAG: keyword score over markdown/jsonl knowledge files."""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DIRS = [
    _ROOT / "knowledge" / "sources",
    _ROOT / "datasets" / "2026",
]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9]{3,}", text.lower()))


def load_chunks(dirs: list[Path] | None = None) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for d in dirs or _DEFAULT_DIRS:
        if not d.exists():
            continue
        for path in d.rglob("*"):
            if path.suffix.lower() not in {".md", ".txt", ".jsonl"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if path.suffix == ".jsonl":
                for i, line in enumerate(text.splitlines()):
                    if line.strip():
                        chunks.append((f"{path.name}:{i}", line[:2000]))
            else:
                for i, para in enumerate(re.split(r"\n\s*\n", text)):
                    para = para.strip()
                    if len(para) > 40:
                        chunks.append((f"{path.name}:{i}", para[:2000]))
    return chunks


def retrieve(query: str, k: int = 4) -> str:
    q = _tokenize(query)
    if not q:
        return ""
    scored: list[tuple[int, str, str]] = []
    for name, chunk in load_chunks():
        overlap = len(q & _tokenize(chunk))
        if overlap:
            scored.append((overlap, name, chunk))
    scored.sort(key=lambda x: -x[0])
    top = scored[:k]
    if not top:
        return ""
    parts = [f"[{name}] {chunk}" for _, name, chunk in top]
    return "\n\n".join(parts)


def inject_context(user_text: str, k: int = 4) -> str:
    ctx = retrieve(user_text, k=k)
    if not ctx:
        return user_text
    return (
        "Usa solo hechos del contexto si aplican. Si no bastan, dilo.\n"
        f"### Contexto\n{ctx}\n\n### Pregunta\n{user_text}"
    )
