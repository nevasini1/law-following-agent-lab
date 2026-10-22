"""Fetch every primary source, extract text, and write a provenance manifest (URL, sha256, timestamp).

    python -m lawful.corpus.build
"""
import datetime as dt
import hashlib
import html
import io
import json
import re

import requests

from lawful.corpus import cap
from lawful.corpus.sources import OPINIONS, STATUTES_AND_REGS
from lawful.paths import CORPUS

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) lawful-research-demo"}


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|nav|header|footer)[^>]*>.*?</\1>", "", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr|section|dd|dt)>", "\n", raw)
    raw = re.sub(r"<[^>]+>", "", raw)
    text = html.unescape(raw)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def pdf_to_text(blob: bytes) -> str:
    from pypdf import PdfReader
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(blob)).pages)


def main():
    manifest = []
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for src in STATUTES_AND_REGS:
        r = requests.get(src.url, headers=UA, timeout=60)
        r.raise_for_status()
        blob = r.content
        ext = "pdf" if src.fmt == "pdf" else "html"
        (CORPUS / "raw" / f"{src.id}.{ext}").write_bytes(blob)
        text = pdf_to_text(blob) if src.fmt == "pdf" else html_to_text(r.text)
        (CORPUS / "text" / f"{src.id}.txt").write_text(text)
        manifest.append({"id": src.id, "title": src.title, "citation": src.citation, "kind": src.kind,
                         "url": src.url, "retrieved_at": now, "sha256": hashlib.sha256(blob).hexdigest(),
                         "chars": len(text)})
        print(f"{src.id:14s} {len(text):>8,d} chars  {src.citation}")

    for key, cite in OPINIONS:
        case = cap.get_case(cite)
        text = cap.opinion_text(case)
        (CORPUS / "raw" / f"case_{key}.json").write_text(json.dumps(case))
        (CORPUS / "text" / f"case_{key}.txt").write_text(text)
        manifest.append({"id": f"case_{key}", "title": case["name_abbreviation"], "citation": cite,
                         "court": case["court"]["name_abbreviation"], "decision_date": case["decision_date"],
                         "kind": "opinion", "url": case["_url"], "retrieved_at": now,
                         "sha256": hashlib.sha256(text.encode()).hexdigest(), "chars": len(text),
                         "cites_to": [c["cite"] for c in case.get("cites_to", [])]})
        print(f"case_{key:9s} {len(text):>8,d} chars  {case['name_abbreviation']}, {cite} ({case['court']['name_abbreviation']} {case['decision_date'][:4]})")

    (CORPUS / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
