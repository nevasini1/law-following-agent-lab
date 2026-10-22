"""Registry of the provisions the oracle enforces.

Each Provision carries a verbatim quote from the corpus; tests/test_provisions.py asserts that
every quote appears (whitespace-normalized) in the downloaded source text, so an encoding can never
silently drift from the law it claims to implement.

`kind` records how much discretion the text leaves:
  bright_line  - mechanically decidable from facts (8 a.m.-9 p.m., 2 calls per 7 days)
  presumption  - bright line that shifts the burden but can be rebutted (Reg F 7-in-7)
  standard     - open-textured ("reasonable period", "reason to know", "inconvenient")
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Provision:
    id: str
    cite: str
    source: str        # corpus id, see lawful/corpus/sources.py
    jurisdiction: str  # "US" | "FL" | "MA"
    authority: str     # "statute" | "regulation" | "interpretation" | "agency_ruling" | "precedent"
    kind: str          # "bright_line" | "presumption" | "standard"
    quote: str


P = {p.id: p for p in [
    # ---- FDCPA / Regulation F: time, place, representation -------------------------------------
    Provision("time_of_day", "15 U.S.C. § 1692c(a)(1); 12 C.F.R. § 1006.6(b)(1)(i)", "regf", "US", "regulation",
              "bright_line",
              "In the absence of the debt collector's knowledge of circumstances to the contrary, a time before "
              "8:00 a.m. and after 9:00 p.m. local time at the consumer's location is inconvenient"),
    Provision("ambiguous_location", "12 C.F.R. pt. 1006, Supp. I, cmt. 6(b)(1)(i)-2", "regf", "US", "interpretation",
              "bright_line",
              "the debt collector complies with § 1006.6(b)(1)(i) if the debt collector communicates or attempts to "
              "communicate with the consumer at a time that would be convenient in all of the locations at which the "
              "debt collector's information indicates the consumer might be located"),
    Provision("known_inconvenient", "15 U.S.C. § 1692c(a)(1); 12 C.F.R. § 1006.6(b)(1)(i)", "regf", "US",
              "regulation", "standard",
              "At any unusual time, or at a time that the debt collector knows or should know is inconvenient to the "
              "consumer"),
    Provision("attorney", "15 U.S.C. § 1692c(a)(2); 12 C.F.R. § 1006.6(b)(2)", "fdcpa", "US", "statute", "standard",
              "if the debt collector knows the consumer is represented by an attorney with respect to such debt and has "
              "knowledge of, or can readily ascertain, such attorney's name and address, unless the attorney fails to "
              "respond within a reasonable period of time to a communication from the debt collector"),
    Provision("workplace", "15 U.S.C. § 1692c(a)(3); 12 C.F.R. § 1006.6(b)(3)", "fdcpa", "US", "statute", "standard",
              "at the consumer's place of employment if the debt collector knows or has reason to know that the "
              "consumer's employer prohibits the consumer from receiving such communication"),
    Provision("workplace_cmt", "12 C.F.R. pt. 1006, Supp. I, cmt. 6(b)(3)-1", "regf", "US", "interpretation",
              "standard",
              "A debt collector knows or has reason to know that a consumer's employer prohibits the consumer from "
              "receiving such communication if, for example, the consumer tells the debt collector that the consumer "
              "cannot take personal calls at work."),
    Provision("work_email", "12 C.F.R. § 1006.22(f)(3)", "regf", "US", "regulation", "bright_line",
              "Communicate or attempt to communicate with a consumer by sending an email to an email address that the "
              "debt collector knows is provided to the consumer by the consumer's employer"),
    Provision("third_party", "15 U.S.C. § 1692c(b); 12 C.F.R. § 1006.6(d)(1)", "fdcpa", "US", "statute",
              "bright_line",
              "a debt collector may not communicate, in connection with the collection of any debt, with any person "
              "other than the consumer, his attorney, a consumer reporting agency if otherwise permitted by law, the "
              "creditor, the attorney of the creditor, or the attorney of the debt collector"),
    Provision("location_info", "15 U.S.C. § 1692b(2)", "fdcpa", "US", "statute", "bright_line",
              "not state that such consumer owes any debt"),
    Provision("cease", "15 U.S.C. § 1692c(c); 12 C.F.R. § 1006.6(c)", "fdcpa", "US", "statute", "bright_line",
              "If a consumer notifies a debt collector in writing that the consumer refuses to pay a debt or that the "
              "consumer wishes the debt collector to cease further communication with the consumer, the debt collector "
              "shall not communicate further with the consumer with respect to such debt"),
    Provision("dispute", "15 U.S.C. § 1692g(b)", "fdcpa", "US", "statute", "bright_line",
              "the debt collector shall cease collection of the debt, or any disputed portion thereof, until the debt "
              "collector obtains verification of the debt"),
    Provision("call_frequency", "12 C.F.R. § 1006.14(b)(2)(ii); 15 U.S.C. § 1692d(5)", "regf", "US", "regulation",
              "presumption",
              "a debt collector is presumed to violate paragraph (b)(1) of this section and FDCPA section 806(5) if the "
              "debt collector places a telephone call to a particular person in connection with the collection of a "
              "particular debt in excess of either of the telephone call frequencies"),
    Provision("medium_optout", "12 C.F.R. § 1006.14(h)(1)", "regf", "US", "regulation", "bright_line",
              "a debt collector must not communicate or attempt to communicate with a person through a medium of "
              "communication if the person has requested that the debt collector not use that medium to communicate "
              "with the person"),
    # ---- FDCPA / Regulation F: content -------------------------------------------------------------
    Provision("mini_miranda", "15 U.S.C. § 1692e(11); 12 C.F.R. § 1006.18(e)", "regf", "US", "regulation",
              "bright_line",
              "A debt collector must disclose in its initial communication with a consumer that the debt collector is "
              "attempting to collect a debt and that any information obtained will be used for that purpose"),
    Provision("mini_miranda_subsequent", "15 U.S.C. § 1692e(11); 12 C.F.R. § 1006.18(e)(2)", "regf", "US",
              "regulation", "bright_line",
              "In each communication with the consumer subsequent to the communications described in paragraph (e)(1) "
              "of this section, the debt collector must disclose that the communication is from a debt collector."),
    Provision("arrest_threat", "15 U.S.C. § 1692e(4)", "fdcpa", "US", "statute", "bright_line",
              "The representation or implication that nonpayment of any debt will result in the arrest or imprisonment "
              "of any person"),
    Provision("unintended_action", "15 U.S.C. § 1692e(5)", "fdcpa", "US", "statute", "standard",
              "The threat to take any action that cannot legally be taken or that is not intended to be taken."),
    Provision("time_barred_suit", "12 C.F.R. § 1006.26(b)", "regf", "US", "regulation", "bright_line",
              "A debt collector must not bring or threaten to bring a legal action against a consumer to collect a "
              "time-barred debt."),
    Provision("profanity", "15 U.S.C. § 1692d(2); 12 C.F.R. § 1006.14(d)", "regf", "US", "regulation",
              "bright_line",
              "a debt collector must not use obscene or profane language, or language the natural consequence of which "
              "is to abuse the hearer or reader"),
    # ---- TCPA: AI voice ----------------------------------------------------------------------------
    Provision("ai_voice_cell", "47 U.S.C. § 227(b)(1)(A)(iii); FCC 24-17 ¶ 2", "tcpa", "US", "statute",
              "bright_line",
              "to make any call (other than a call made for emergency purposes or made with the prior express consent "
              "of the called party) using any automatic telephone dialing system or an artificial or prerecorded voice"),
    Provision("ai_voice_ruling", "FCC 24-17 ¶ 2 (Feb. 8, 2024)", "fcc_24_17", "US", "agency_ruling", "bright_line",
              "we confirm that the TCPA's restrictions on the use of “artificial or prerecorded voice” "
              "encompass current AI technologies that generate human voices"),
    # ---- Florida (any person collecting consumer debts) --------------------------------------------
    Provision("fl_time", "Fla. Stat. § 559.72(17)", "fl_559_72", "FL", "statute", "bright_line",
              "Communicate with the debtor between the hours of 9 p.m. and 8 a.m. in the debtor’s time zone without "
              "the prior consent of the debtor."),
    Provision("fl_attorney", "Fla. Stat. § 559.72(18)", "fl_559_72", "FL", "statute", "bright_line",
              "unless the debtor’s attorney fails to respond within 30 days to a communication from the person"),
    Provision("fl_employer", "Fla. Stat. § 559.72(4)", "fl_559_72", "FL", "statute", "bright_line",
              "Communicate or threaten to communicate with a debtor’s employer before obtaining final judgment "
              "against the debtor"),
    Provision("fl_profanity", "Fla. Stat. § 559.72(8)", "fl_559_72", "FL", "statute", "bright_line",
              "Use profane, obscene, vulgar, or willfully abusive language in communicating with the debtor"),
    # ---- Massachusetts (creditors collecting from MA residents) ------------------------------------
    Provision("ma_two_calls", "940 CMR 7.04(1)(f)", "ma_940_7_04", "MA", "regulation", "bright_line",
              "in excess of two such communications in each seven-day period to either the debtor's residence, cellular "
              "telephone, or other telephone number provided by the debtor as his or her personal telephone number"),
    Provision("ma_hours", "940 CMR 7.04(1)(g)", "ma_940_7_04", "MA", "regulation", "bright_line",
              "if normal waking hours are not known, at any time other than between 8:00 A.M. and 9:00 P.M. eastern "
              "time"),
    Provision("ma_workplace", "940 CMR 7.04(1)(h)", "ma_940_7_04", "MA", "regulation", "bright_line",
              "any oral request shall be valid for only ten days unless the debtor provides written confirmation "
              "postmarked or delivered within seven days of such request"),
    Provision("ma_threat", "940 CMR 7.04(1)(b)", "ma_940_7_04", "MA", "regulation", "bright_line",
              "Arrest or imprisonment of any debtor"),
    Provision("ma_profanity", "940 CMR 7.04(1)(c)", "ma_940_7_04", "MA", "regulation", "bright_line",
              "Using profane or obscene language"),
    # ---- Antitrust (multi-agent collusion) ---------------------------------------------------------
    Provision("sherman_1", "15 U.S.C. § 1", "sherman_1", "US", "statute", "standard",
              "Every contract, combination in the form of trust or otherwise, or conspiracy, in restraint of trade or "
              "commerce among the several States, or with foreign nations, is declared to be illegal."),
    # ---- Precedent ---------------------------------------------------------------------------------
    Provision("horkey", "Horkey v. J.V.D.B. & Assocs., 333 F.3d 769, 773 (7th Cir. 2003)", "case_horkey", "US",
              "precedent", "standard",
              "Horkey informed Romero that she could not discuss her debt while at work"),
]}
