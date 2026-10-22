"""A small defeasible deontic engine for conflicting obligations across overlapping sources.

Norms say an act is Forbidden, Obligatory or Permitted under a condition. When applicable norms disagree,
the conflict is resolved -- and the resolution is *explained* -- by an ordered list of principles:

  1. exception      a norm that names another as the rule it carves out from defeats it (1692c(c)(1) vs 1692c(c))
  2. lex superior   statute/regulation > agency guidance > private instruction
  3. preemption     a jurisdiction-specific clause, e.g. 15 U.S.C. § 1692n: a state law affording consumers
                    *greater* protection is not preempted, so between federal and state law the more
                    protective (restrictive) norm controls
  4. lex specialis  the norm with the more specific condition wins
  5. lex posterior  the later-enacted norm wins
Unresolved conflicts are reported as such -- the right output for an agent is then to escalate, not guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

RANK = {"constitution": 4, "statute": 3, "regulation": 3, "interpretation": 2, "agency_ruling": 2,
        "instruction": 0}


@dataclass(frozen=True)
class Norm:
    id: str
    modality: str                    # "F" forbidden | "O" obligatory | "P" permitted
    act: str                         # act type the norm speaks to
    when: Callable[[dict], bool]     # condition over facts
    source: str                      # citation or "manager instruction"
    authority: str                   # key into RANK
    jurisdiction: str = "US"
    specificity: int = 0             # number of conditions; higher = more specific
    enacted: int = 0                 # year
    excepts: tuple[str, ...] = ()    # ids of norms this one carves an exception from
    protective: bool = False         # restricts the collector in the consumer's favor


@dataclass
class Resolution:
    act: str
    verdict: str                     # "forbidden" | "obligatory" | "permitted" | "unresolved" | "no norm"
    applicable: list[str]
    trace: list[str] = field(default_factory=list)
    defeats: list[tuple[str, str, str]] = field(default_factory=list)  # (winner source, loser source, principle)
    sources: list[str] = field(default_factory=list)


def _beats(a: Norm, b: Norm) -> str | None:
    """Return the principle under which norm a defeats norm b, or None."""
    if b.id in a.excepts:
        return "exception"
    if RANK[a.authority] != RANK[b.authority]:
        return "lex superior" if RANK[a.authority] > RANK[b.authority] else None
    if {a.jurisdiction, b.jurisdiction} >= {"US"} and a.jurisdiction != b.jurisdiction and a.protective != b.protective:
        return "preemption (§ 1692n: more protective law controls)" if a.protective else None
    if a.specificity != b.specificity:
        return "lex specialis" if a.specificity > b.specificity else None
    if a.enacted != b.enacted:
        return "lex posterior" if a.enacted > b.enacted else None
    return None


def resolve(norms: list[Norm], act: str, facts: dict) -> Resolution:
    live = [n for n in norms if n.act == act and n.when(facts)]
    res = Resolution(act, "no norm", [n.id for n in live], sources=[n.source for n in live])
    if not live:
        return res
    # a norm survives if no conflicting norm defeats it (F conflicts with P and with O; O and P are compatible)
    survivors = []
    for n in live:
        defeated = False
        for m in live:
            if m is n or "F" not in {m.modality, n.modality} or m.modality == n.modality:
                continue
            why = _beats(m, n)
            if why:
                res.trace.append(f"{m.id} [{m.modality}, {m.source}] defeats {n.id} [{n.modality}, {n.source}] by {why}")
                res.defeats.append((m.source, n.source, why))
                defeated = True
        if not defeated:
            survivors.append(n)
    mods = {n.modality for n in survivors}
    legal_duty = any(n.modality == "O" and n.authority != "instruction" for n in survivors)
    if "F" in mods and mods & {"P", "O"}:
        res.verdict = "unresolved"
        res.trace.append("unresolved conflict among: " + ", ".join(f"{n.id}[{n.modality}]" for n in survivors))
    elif "F" in mods:
        res.verdict = "forbidden"
    elif legal_duty:
        res.verdict = "obligatory"
    else:
        res.verdict = "permitted" + (" (instructed)" if "O" in mods else "")
    return res


# ------------------------------------------------------------------ the debt-collection norm base
def debt_collection_norms() -> list[Norm]:
    third_party = lambda f: f.get("collector") == "third_party" and f.get("consumer_debt", True)
    return [
        Norm("manager.call_now", "O", "call_now", lambda f: f.get("manager_says_call"), "manager instruction",
             "instruction"),
        Norm("regf.1006.6(b)(1)(i)", "F", "call_now",
             lambda f: third_party(f) and not f.get("all_locations_convenient", True),
             "12 C.F.R. § 1006.6(b)(1)(i) + cmt. 6(b)(1)(i)-2", "regulation", specificity=2, enacted=2021,
             protective=True),
        Norm("regf.1006.6(b)(4)(i)", "P", "call_now", lambda f: third_party(f) and f.get("consumer_consented"),
             "12 C.F.R. § 1006.6(b)(4)(i) (prior consent)", "regulation", specificity=3, enacted=2021,
             excepts=("regf.1006.6(b)(1)(i)",)),
        Norm("fl.559.72(17)(a)", "P", "call_now",
             lambda f: f.get("state") == "FL" and f.get("area_code_tz_convenient", True),
             "Fla. Stat. § 559.72(17)(a) (area-code presumption)", "statute", jurisdiction="FL", specificity=2,
             enacted=2010),
        Norm("fl.559.72(17)", "F", "call_now",
             lambda f: f.get("state") == "FL" and not f.get("area_code_tz_convenient", True)
             and not f.get("consumer_consented"),
             "Fla. Stat. § 559.72(17)", "statute", jurisdiction="FL", specificity=2, enacted=2010, protective=True),
        Norm("fdcpa.1692c(c)", "F", "send_letter", lambda f: third_party(f) and f.get("cease_letter"),
             "15 U.S.C. § 1692c(c)", "statute", specificity=1, enacted=1977, protective=True),
        Norm("fdcpa.1692c(c)(1)", "P", "send_letter",
             lambda f: third_party(f) and f.get("cease_letter") and f.get("letter_is_termination_notice"),
             "15 U.S.C. § 1692c(c)(1)", "statute", specificity=2, enacted=1977, excepts=("fdcpa.1692c(c)",)),
        Norm("ma.7.04(1)(h)", "F", "call_work",
             lambda f: f.get("state") == "MA" and f.get("collector") == "creditor" and f.get("ma_request_in_force"),
             "940 CMR 7.04(1)(h)", "regulation", jurisdiction="MA", specificity=2, protective=True),
        Norm("fdcpa.1692c(a)(3)", "F", "call_work", lambda f: third_party(f) and f.get("reason_to_know_employer"),
             "15 U.S.C. § 1692c(a)(3); Horkey, 333 F.3d 769", "statute", specificity=1, enacted=1977,
             protective=True),
    ]


WORKED_CONFLICTS = [
    ("Manager says call now; it is 7:20 a.m. at one indicated location (third-party collector)", "call_now",
     dict(manager_says_call=True, collector="third_party", all_locations_convenient=False)),
    ("Same, but the consumer gave prior consent to be called at this hour", "call_now",
     dict(manager_says_call=True, collector="third_party", all_locations_convenient=False, consumer_consented=True)),
    ("Florida consumer; area-code time zone is fine but the address time zone is 7:20 a.m. (third-party collector)",
     "call_now", dict(collector="third_party", state="FL", area_code_tz_convenient=True,
                      all_locations_convenient=False)),
    ("Same facts, but the caller is the original creditor (FDCPA inapplicable)", "call_now",
     dict(collector="creditor", state="FL", area_code_tz_convenient=True, all_locations_convenient=False)),
    ("Cease letter on file; proposed letter only says collection efforts are terminated", "send_letter",
     dict(collector="third_party", cease_letter=True, letter_is_termination_notice=True)),
    ("Cease letter on file; proposed letter asks for payment", "send_letter",
     dict(collector="third_party", cease_letter=True, letter_is_termination_notice=False)),
    ("MA creditor; oral 'don't call me at work' 14 days ago, never confirmed in writing", "call_work",
     dict(collector="creditor", state="MA", ma_request_in_force=False)),
    ("Same words to a third-party collector: under federal law they give reason to know (Horkey)", "call_work",
     dict(collector="third_party", state="MA", reason_to_know_employer=True)),
]
