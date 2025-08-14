# citations.py
from typing import List, Tuple, Dict

def _line(meta: Dict, fallback: str) -> str:
    authors = meta.get("authors")
    title   = meta.get("paper_title") or meta.get("title")
    year    = meta.get("year")
    if not (authors or title):
        return fallback
    parts = []
    if authors: parts.append(f"{authors}")
    if year:    parts.append(f"({year}).")
    if title:   parts.append(f"{title}.")
    return " ".join(parts).strip()

def build_references_block(ctx: List[Tuple[str, dict]]) -> str:
    """ctx: list of (doc_text, metadata_dict) from retrieval"""
    seen = set()
    lines = []
    for _, m in ctx:
        key = (m.get("authors"), m.get("paper_title") or m.get("title"), m.get("year"))
        if key in seen:
            continue
        seen.add(key)
        fallback = m.get("file_name") or m.get("source_filename") or "Unknown source"
        lines.append(f"- {_line(m, fallback)}")
    return "\n\n**References**\n" + "\n".join(lines) if lines else ""
