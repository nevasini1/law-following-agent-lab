"""Minimal client for the Caselaw Access Project static export (https://static.case.law).

Layout: /{reporter_slug}/{volume}/CasesMetadata.json lists every case in a volume;
/{reporter_slug}/{volume}/cases/{first_page:04d}-{nn}.json holds the full opinion text.
"""
import functools
import re

import requests

BASE = "https://static.case.law"
REPORTER_SLUGS = {"F.2d": "f2d", "F.3d": "f3d", "F.4th": "f4th", "F. Supp.": "f-supp",
                  "F. Supp. 2d": "f-supp-2d", "F. Supp. 3d": "f-supp-3d", "U.S.": "us"}
_session = requests.Session()
_session.headers["User-Agent"] = "lawful-research-demo"


def parse_cite(cite: str) -> tuple[str, int, int]:
    m = re.fullmatch(r"(\d+)\s+(.+?)\s+(\d+)", cite.strip())
    if not m:
        raise ValueError(f"unparseable citation: {cite}")
    vol, reporter, page = m.groups()
    reporter = reporter.replace("F.Supp.", "F. Supp.").replace("F. Supp.2d", "F. Supp. 2d")
    return REPORTER_SLUGS[reporter], int(vol), int(page)


@functools.lru_cache(maxsize=256)
def volume_metadata(slug: str, vol: int) -> list[dict]:
    r = _session.get(f"{BASE}/{slug}/{vol}/CasesMetadata.json", timeout=60)
    r.raise_for_status()
    return r.json()


def get_case(cite: str) -> dict:
    """Return the full CAP case record (metadata + casebody) for a reporter citation."""
    slug, vol, page = parse_cite(cite)
    # a few records carry page ranges ("1103-1113") in first_page
    meta = [c for c in volume_metadata(slug, vol) if str(c["first_page"]).split("-")[0] == str(page)]
    if not meta:
        raise LookupError(f"{cite} not found in CAP volume {slug}/{vol}")
    path = f"{BASE}/{slug}/{vol}/cases/{meta[0]['file_name']}.json"
    r = _session.get(path, timeout=60)
    r.raise_for_status()
    case = r.json()
    case["_url"] = path
    return case


def opinion_text(case: dict) -> str:
    ops = case["casebody"]["opinions"]
    return "\n\n".join(f"[{o.get('type', 'opinion').upper()} — {o.get('author') or 'per curiam'}]\n{o['text']}" for o in ops)
