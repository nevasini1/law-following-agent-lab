"""Repo-relative paths. Everything an experiment writes lands under results/."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CORPUS = DATA / "corpus"
SCENARIOS = DATA / "scenarios"
RESULTS = ROOT / "results"

for _p in (CORPUS / "raw", CORPUS / "text", SCENARIOS, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)
