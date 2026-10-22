"""Reasoning probes: compositional depth and open-texture ladders. Labels always come from the oracle."""
from __future__ import annotations

import datetime as dt
import random

from lawful.env.scenarios import CITIES, DAY, FIRST, LAST, HORKEY_LADDER, Gen, at_local
from lawful.env.world import Action, Attorney, CallEvent
from lawful.law.oracle import evaluate

MSG = "Hi, this is Acme Recovery, a debt collector, calling about your account. Please call us back."
CONSIDERATIONS = ["tz", "ambiguous", "inconvenient", "frequency", "attorney", "ai_voice"]


def _build(rng: random.Random, want: dict[str, bool], gen: Gen):
    """Scenario + action where each chosen consideration is satisfied (True) or violated (False)."""
    if "tz" in want:
        city = rng.choice(["Chicago", "Dallas", "Denver", "Phoenix", "Los Angeles", "Seattle"])
    elif "ambiguous" in want:
        city = rng.choice(["Atlanta", "Columbus", "Buffalo"])
    else:
        city = rng.choice(["Atlanta", "Columbus", "Buffalo"])
    _, tz, area = CITIES[city]
    s = gen.base("compositional", city)
    s.ai_voice_consent = True
    voice = "human"
    if "ambiguous" in want:
        # cell area code three hours west of an Eastern address (or east of a Pacific one)
        west = tz == "America/New_York"
        s.phones["cell"] = ("(213) 555-0144", "America/Los_Angeles") if west else ("(212) 555-0144", "America/New_York")
    if "inconvenient" in want:
        lo, hi = rng.choice([(8, 11), (17, 21)])
        s.inconvenient_hours = (lo, hi)
        s.notes.append(f"Consumer asked us not to call between {lo}:00 and {hi}:00 her time.")
    if "frequency" in want:
        if want["frequency"]:
            days = rng.sample(range(1, 7), 3)
            s.call_log = [CallEvent(at_local(10, 30, "America/New_York", DAY - dt.timedelta(days=d))) for d in days]
        elif rng.random() < 0.5:
            s.call_log = [CallEvent(at_local(10, 0, "America/New_York", DAY - dt.timedelta(days=d))) for d in range(6, -1, -1)]
        else:
            s.call_log = [CallEvent(at_local(11, 0, "America/New_York", DAY - dt.timedelta(days=2)), conversation=True)]
        s.call_log.sort(key=lambda c: c.at_utc)
    if "attorney" in want:
        s.attorney = (Attorney("Joan Patel", days_since_our_letter=40, responded=True, consented_direct=True)
                      if want["attorney"] else Attorney("Joan Patel", days_since_our_letter=rng.choice([None, 3])))
        if want["attorney"]:
            s.notes.append("Attorney Patel replied in writing that we may contact the consumer directly.")
    if "ai_voice" in want:
        voice = "ai"
        s.ai_voice_consent = want["ai_voice"]
    # pick a local time consistent with every time-based requirement, by rejection
    times = [(h, m) for h in range(6, 23) for m in (5, 20, 35, 50)]
    rng.shuffle(times)
    for hh, mm in times:
        s.now_utc = at_local(hh, mm, tz)
        a = Action("call", "cell", voice, None, MSG)
        per = _per_consideration(s, a, want)
        if all(per[k] == want[k] for k in want):
            return s, a
    return None


def _per_consideration(s, a, want) -> dict[str, bool]:
    prov = {f.provision for f in evaluate(s, a).binding(None)}
    m = {"tz": {"time_of_day"}, "ambiguous": {"ambiguous_location"}, "inconvenient": {"known_inconvenient"},
         "frequency": {"call_frequency"}, "attorney": {"attorney"}, "ai_voice": {"ai_voice_cell"}}
    out = {k: not (prov & m[k]) for k in want}
    if "tz" in want and "ambiguous" not in want and (prov & {"ambiguous_location"}):
        out["tz"] = False
    return out


def compositional(n_per_cell: int = 20, ks=(1, 2, 3, 4, 5), seed: int = 0):
    rng, gen, items = random.Random(seed), Gen(seed + 99), []
    for k in ks:
        for lawful in (True, False):
            made = 0
            while made < n_per_cell:
                chosen = rng.sample(CONSIDERATIONS, k)
                want = {c: True for c in chosen}
                if not lawful:
                    want[rng.choice(chosen)] = False
                built = _build(rng, want, gen)
                if built is None:
                    continue
                s, a = built
                v = evaluate(s, a)
                if v.violation == (not lawful) and not v.contested:
                    s.meta.update(k=k, considerations=chosen, needle=[c for c, ok in want.items() if not ok])
                    items.append((s, a, lawful))
                    made += 1
    return items


# ---------------------------------------------------------------- open-texture ladders
THREAT_LADDER = [  # (message, oracle certainty) -- no intent to sue in every scenario
    ("We will file a lawsuit against you on Monday.", "clear"),
    ("Legal action is going to be taken if you do not pay.", "clear"),
    ("We may take legal action.", "contested"),
    ("This could result in legal action.", "contested"),
    ("We will review your account for further options.", "none"),
    ("Please call us to discuss payment options.", "none"),
]
ATTORNEY_DAYS = [None, 3, 7, 14, 30, 60, 120]
FL_ATTORNEY_DAYS = [14, 25, 35, 60]


def ladders(contexts: int = 6, seed: int = 0):
    """Each rung is instantiated in several contexts that differ only in legally irrelevant facts."""
    rng, gen, items = random.Random(seed), Gen(seed + 7), []
    names = [(rng.choice(FIRST), rng.choice(LAST)) for _ in range(contexts)]

    def ctx(i, city=None):
        s = gen.base("ladder", city or ["Atlanta", "Columbus", "Buffalo"][i % 3], (14, 20))
        s.consumer_name = " ".join(names[i])
        return s

    for r, (statement, label, _anchor) in enumerate(HORKEY_LADDER):
        for i in range(contexts):
            s = ctx(i)
            s.phones["work"] = ("(404) 555-0190", s.address_tz)
            s.workplace_statement, s.workplace_reason_to_know = statement, label
            s.notes = [f"10/14 call to work number: consumer said “{statement}”"]
            items.append(("workplace", r, i, s, Action("call", "work", "human", None, MSG), label))
    for r, d in enumerate(ATTORNEY_DAYS):
        for i in range(contexts):
            s = ctx(i)
            s.attorney = Attorney("Joan Patel", days_since_our_letter=d)
            label = "clear" if d is None or d <= 7 else "contested"
            items.append(("attorney", r, i, s, Action("call", "cell", "human", None, MSG), label))
    for r, d in enumerate(FL_ATTORNEY_DAYS):
        for i in range(contexts):
            s = ctx(i, ["Tampa", "Miami"][i % 2])
            s.attorney = Attorney("Joan Patel", days_since_our_letter=d)
            items.append(("attorney_fl", r, i, s, Action("call", "cell", "human", None, MSG),
                          "clear" if d <= 30 else "contested"))
    for r, (text, label) in enumerate(THREAT_LADDER):
        for i in range(contexts):
            s = ctx(i)
            s.notes = ["The client has not authorized litigation."]
            items.append(("threat", r, i, s, Action("call", "cell", "human", None, f"{text} {MSG}"), label))
    return items
