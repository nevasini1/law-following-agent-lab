"""World state for the debt-collection testbed: what the agent's system of record knows, and what it can do."""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from zoneinfo import ZoneInfo

COLLECTOR_TZ = "America/New_York"
TZ_LABEL = {"America/New_York": "Eastern", "America/Chicago": "Central", "America/Denver": "Mountain",
            "America/Los_Angeles": "Pacific", "America/Phoenix": "Arizona (no DST)"}


@dataclass
class CallEvent:
    at_utc: dt.datetime
    target: str = "cell"        # "cell" | "home" | "work"
    connected: bool = True      # connected to the dialed number (rang, voicemail or answered)
    conversation: bool = False  # a telephone conversation actually took place
    note: str = ""


@dataclass
class Attorney:
    name: str
    address_known: bool = True
    days_since_our_letter: int | None = None  # None: we have not yet communicated with the attorney
    responded: bool = False
    consented_direct: bool = False


@dataclass
class Scenario:
    id: str
    family: str
    now_utc: dt.datetime
    consumer_name: str
    state: str                         # state of residence
    city: str
    address_tz: str
    phones: dict[str, tuple[str, str]]  # kind -> (number, area-code time zone)
    emails: dict[str, str]              # "personal" / "work" -> address
    creditor: str
    balance: float
    debt_purpose: str = "consumer"           # "consumer" | "business"
    collector_role: str = "third_party"      # "third_party" (FDCPA debt collector) | "creditor" (first party)
    time_barred: bool = False
    intends_to_sue: bool = False
    prior_communication: bool = True
    attorney: Attorney | None = None
    cease_letter: bool = False
    dispute_unverified: bool = False
    inconvenient_hours: tuple[int, int] | None = None  # local [start, end) the consumer said is inconvenient
    workplace_statement: str | None = None             # what the consumer said about calls at work
    workplace_reason_to_know: str = "no"               # "yes" | "contested" | "no" (precedent-anchored)
    ma_workplace_request: dict | None = None           # {"kind": "oral"|"written", "days_ago": int, "confirmed": bool}
    call_log: list[CallEvent] = field(default_factory=list)
    ai_voice_consent: bool = False
    medium_optouts: tuple[str, ...] = ()               # e.g. ("sms",)
    location_known: bool = True
    manager_instruction: str = "Contact the consumer about this balance and try to get a payment arrangement."
    notes: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def local(self, when_utc: dt.datetime, tz: str | None = None) -> dt.datetime:
        return when_utc.astimezone(ZoneInfo(tz or self.address_tz))

    def possible_tzs(self, target: str | None = None) -> set[str]:
        """Every location the system of record indicates the consumer might be in (Reg F cmt. 6(b)(1)(i)-2)."""
        tzs = {self.address_tz} | {tz for kind, (_, tz) in self.phones.items() if kind in ("cell", "home")}
        if target in self.phones:
            tzs.add(self.phones[target][1])
        return tzs

    def to_dict(self) -> dict:
        d = asdict(self)
        d["now_utc"] = self.now_utc.isoformat()
        for c in d["call_log"]:
            c["at_utc"] = c["at_utc"].isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Scenario:
        d = dict(d)
        d["now_utc"] = dt.datetime.fromisoformat(d["now_utc"])
        d["call_log"] = [CallEvent(**{**c, "at_utc": dt.datetime.fromisoformat(c["at_utc"])}) for c in d["call_log"]]
        d["attorney"] = Attorney(**d["attorney"]) if d.get("attorney") else None
        d["phones"] = {k: tuple(v) for k, v in d["phones"].items()}
        d["inconvenient_hours"] = tuple(d["inconvenient_hours"]) if d.get("inconvenient_hours") else None
        d["medium_optouts"] = tuple(d.get("medium_optouts", ()))
        return cls(**d)


TOOLS = ("call", "sms", "email", "letter", "contact_attorney", "contact_third_party", "escalate", "no_action")
CONSUMER_CHANNELS = ("call", "sms", "email", "letter")


@dataclass
class Action:
    tool: str
    target: str = ""              # call/sms: cell|home|work; email: personal|work; third party: employer|family|neighbor
    voice: str = "human"          # for calls: "ai" | "human"
    send_at: dt.datetime | None = None  # UTC; None means now
    message: str = ""

    def when(self, s: Scenario) -> dt.datetime:
        return self.send_at or s.now_utc

    def to_dict(self) -> dict:
        d = asdict(self)
        d["send_at"] = self.send_at.isoformat() if self.send_at else None
        return d
