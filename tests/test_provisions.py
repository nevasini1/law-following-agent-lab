"""Every provision the oracle enforces must quote its source verbatim."""
import re

import pytest

from lawful.law.provisions import P
from lawful.paths import CORPUS


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


@pytest.mark.parametrize("pid", sorted(P))
def test_quote_is_verbatim(pid):
    prov = P[pid]
    text = _norm((CORPUS / "text" / f"{prov.source}.txt").read_text())
    assert _norm(prov.quote) in text, f"{pid}: quote not found in {prov.source}"
