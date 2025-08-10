# metadata.py
import json, os, re, hashlib
from datetime import date

def parse_filename(fname: str):
    # expected: title_author_publisheddate_journal.pdf (flexible)
    base = os.path.basename(fname)
    stem, _ = os.path.splitext(base)
    parts = stem.split("_")
    meta = {"title": None, "authors": [], "published_date": None, "journal": None}
    if len(parts) >= 4:
        meta["title"] = parts[0].replace("-", " ")
        meta["authors"] = [a.replace("-", " ") for a in parts[1].split("+")]  # allow First-Last+Second-Last
        # try YYYY-MM-DD, YYYY-MM, or YYYY
        d = parts[2]
        if re.match(r"^\d{4}-\d{2}-\d{2}$", d): meta["published_date"] = d
        elif re.match(r"^\d{4}-\d{2}$", d):     meta["published_date"] = d + "-01"
        elif re.match(r"^\d{4}$", d):           meta["published_date"] = d + "-01-01"
        meta["journal"] = parts[3].replace("-", " ")
    return meta

def sha256(path: str):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def sidecar_path(pdf_path: str) -> str:
    root, _ = os.path.splitext(pdf_path)
    return root + ".json"

def load_or_init(pdf_path: str, filename_fallback=True) -> dict:
    jpath = sidecar_path(pdf_path)
    if os.path.exists(jpath):
        with open(jpath, "r", encoding="utf-8") as f:
            return json.load(f)

    meta = {
        "doc_id": None,
        "title": None, "authors": [], "published_date": None, "journal": None,
        "doi": None, "url": None,
        "volume": None, "issue": None, "pages": None,
        "abstract": None,
        "keywords": [],
        "pdf_path": pdf_path,
        "hash": sha256(pdf_path),
        "reference_style_cache": {},   # {"apa": "..."}
        "ingested_at": date.today().isoformat()
    }
    if filename_fallback:
        meta.update({k:v for k,v in parse_filename(pdf_path).items() if v})
    # stable citekey
    year = (meta["published_date"] or "xxxx")[:4]
    first = (meta["authors"][0] if meta["authors"] else "unknown").split()[-1].title() if meta["authors"] else "Unknown"
    short = (meta["title"] or os.path.basename(pdf_path)).split()[0].title()
    meta["doc_id"] = f"{first}_{year}_{short}"

    return meta

def save_sidecar(pdf_path: str, meta: dict):
    with open(sidecar_path(pdf_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def apa_string(meta: dict) -> str:
    # ultra-simple APA (good enough until you add citeproc)
    authors = meta.get("authors") or []
    if authors:
        if len(authors) == 1:
            a = authors[0]
        elif len(authors) == 2:
            a = f"{authors[0]} & {authors[1]}"
        else:
            a = f"{', '.join(authors[:-1])}, & {authors[-1]}"
    else:
        a = ""
    year = (meta.get("published_date") or "")[:4]
    title = meta.get("title") or ""
    journal = meta.get("journal") or ""
    vol = meta.get("volume") or ""
    iss = f"({meta.get('issue')})" if meta.get("issue") else ""
    pages = meta.get("pages") or ""
    doi = f"https://doi.org/{meta['doi']}" if meta.get("doi") else (meta.get("url") or "")
    pieces = [p for p in [a, f"({year}).", title + ".", journal, vol + iss + ("," if pages else ""), pages + ".", doi] if p]
    return " ".join(pieces).replace(" ,", ",")

def ensure_reference_cache(meta: dict) -> dict:
    if not meta.get("reference_style_cache", {}).get("apa"):
        meta.setdefault("reference_style_cache", {})["apa"] = apa_string(meta)
    return meta
