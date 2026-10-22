"""Primary legal sources. Every rule the oracle enforces must trace back to one of these.

Statutes and regulations come from official publishers (GovInfo, eCFR, state legislatures);
court opinions come from the Caselaw Access Project's static bulk export (no API key needed).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    citation: str
    url: str
    kind: str  # "statute" | "regulation" | "agency_ruling" | "opinion"
    fmt: str = "html"  # "html" | "pdf" | "cap_json"


STATUTES_AND_REGS = [
    Source("fdcpa", "Fair Debt Collection Practices Act", "15 U.S.C. §§ 1692-1692p",
           "https://www.govinfo.gov/content/pkg/USCODE-2023-title15/html/USCODE-2023-title15-chap41-subchapV.htm",
           "statute"),
    Source("regf", "Regulation F (Debt Collection Practices)", "12 C.F.R. pt. 1006",
           "https://www.ecfr.gov/api/renderer/v1/content/enhanced/current/title-12?part=1006", "regulation"),
    Source("tcpa", "Telephone Consumer Protection Act", "47 U.S.C. § 227",
           "https://www.govinfo.gov/content/pkg/USCODE-2023-title47/html/USCODE-2023-title47-chap5-subchapII-partI-sec227.htm",
           "statute"),
    Source("tcpa_rules", "FCC TCPA rules: delivery restrictions", "47 C.F.R. § 64.1200",
           "https://www.ecfr.gov/api/renderer/v1/content/enhanced/current/title-47?part=64&section=64.1200",
           "regulation"),
    Source("fcc_24_17", "FCC Declaratory Ruling: AI-generated voices are 'artificial' under the TCPA",
           "FCC 24-17, CG Docket No. 23-362 (Feb. 8, 2024)",
           "https://docs.fcc.gov/public/attachments/FCC-24-17A1.pdf", "agency_ruling", "pdf"),
    Source("fl_559_72", "Florida Consumer Collection Practices Act: prohibited practices", "Fla. Stat. § 559.72",
           "http://www.leg.state.fl.us/statutes/index.cfm?App_mode=Display_Statute&URL=0500-0599/0559/Sections/0559.72.html",
           "statute"),
    Source("sherman_1", "Sherman Act \u00a7 1", "15 U.S.C. \u00a7 1",
           "https://www.govinfo.gov/content/pkg/USCODE-2023-title15/html/USCODE-2023-title15-chap1-sec1.htm",
           "statute"),
    Source("ma_940_7_03", "Massachusetts debt collection regulations: definitions", "940 CMR 7.03",
           "https://www.law.cornell.edu/regulations/massachusetts/940-CMR-7-03", "regulation"),
    Source("ma_940_7_04", "Massachusetts debt collection regulations: contact with debtors", "940 CMR 7.04",
           "https://www.law.cornell.edu/regulations/massachusetts/940-CMR-7-04", "regulation"),
]

# Seed opinions, addressed by reporter citation in the Caselaw Access Project static export.
OPINIONS = [
    ("horkey", "333 F.3d 769"),     # workplace calls after "I can't talk at work"; profanity (7th Cir. 2003)
    ("jeter", "760 F.2d 1168"),     # least-sophisticated-consumer standard; threat of suit (11th Cir. 1985)
    ("clomon", "988 F.2d 1314"),    # least-sophisticated-consumer standard; attorney letters (2d Cir. 1993)
    ("brown_csc", "464 F.3d 450"),  # threats of action not intended, 1692e(5) (3d Cir. 2006)
    ("hahn", "557 F.3d 755"),       # materiality requirement under 1692e (7th Cir. 2009)
    ("donohue", "592 F.3d 1027"),   # materiality requirement under 1692e (9th Cir. 2010)
    ("gammon", "27 F.3d 1254"),     # "unsophisticated consumer" formulation (7th Cir. 1994)
]
