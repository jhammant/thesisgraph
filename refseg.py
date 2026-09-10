#!/usr/bin/env python3
"""refseg.py — split a reference section into individual reference entries.

Segmentation, not parsing. GROBID parses an entry into author/title/year very
well but expects one entry per input; the corpus stores the reference section as
run-on text with line structure already discarded. Recovering the boundaries is
therefore the binding constraint on the citation graph, and it is what this
module does. Parsing is left to GROBID.

Measured against the previous author-date-only parser, the styles below are the
ones that actually occur in the corpus and the ones it could not see:

    [12] J. D. Lawson. Some criteria...        numeric, bracketed   (STEM)
    (1) Alberts, B.; Johnson, A. ...           numeric, parens      (ACS)
    12. Smith, J. Title...                     numeric, bare
    - Abbasi, T. and Abbasi, S.A., 2007.       bulleted             (engineering)
    Abascal J, Nicolle C (2005). Moving...     author-date, initial-first
    Braun, V. and Clarke, V. (2006). Using...  author-date, classic Harvard

A reference section is not always a reference list. History theses list archival
sources (record-office catalogue numbers, with no author, year or title), and
the `references` bucket sometimes captures a data table outright. Both are
reported by `looks_bibliographic` rather than counted as parse failures, because
treating them as misses overstates how much of the corpus a parser can recover.
"""
from __future__ import annotations
import re

# --- entry-start markers, most specific first -------------------------------
BRACKET = re.compile(r"\[(\d{1,3})\]")
PARENNUM = re.compile(r"(?:(?<=\s)|^)\((\d{1,3})\)\s+(?=[A-Z])")
BARENUM = re.compile(r"(?:(?<=\s)|^)(\d{1,3})\.\s+(?=[A-Z][a-z])")
BULLET = re.compile(r"(?:(?<=\s)|^)[-–•]\s+(?=[A-Z])")

# An author-date entry opens with a surname followed by initials, in either
# order: "Smith, J." or "Smith J,". Requiring a preceding sentence end is what
# made the old parser miss bulleted and run-on lists, so the boundary is the
# surname pattern itself.
# \w with re.UNICODE, not [A-Za-z]: transliterated surnames carrying macrons
# and dots below ("'Abadi, Muhammad Shams al-Haqq") are ordinary in law and area
# studies, and an ASCII class cannot see them at all.
_SUR = r"[^\W\d_][\w'’\-]{1,24}"
# The boundary may follow a digit as well as a full stop: an entry ending in a
# page range ("399-421 Ahmed, S. (2004)") has no punctuation before the next
# author at all. The third alternative below is the humanities style, which
# spells first names out ("Abadi, Muhammad Shams") instead of initialising them.
AUTHDATE = re.compile(
    r"(?<=[.\]\)’\"\d])\s+"
    r"(?=" + _SUR + r",?\s*(?:[^\W\d_]\.){1,4}|"          # Smith, J.J.
           + _SUR + r"\s+[^\W\d_][,\s]*\(?\d{4}|"          # Bell J (2010)
           + _SUR + r"\s+[^\W\d_],|"                       # Abascal J, Nicolle
           + _SUR + r"(?:\s+" + _SUR + r")?,\s+"            # Abu Dawud, Sulayman
           + r"[^\W\d_][\w'’\-]{2,24}[,\s])",
    re.UNICODE)

YEAR = re.compile(r"\b(1[5-9]\d{2}|20[0-4]\d)\b")
TRAILING_PAGES = re.compile(r"\s+\d{1,3}(?:\s*,\s*\d{1,3})*\s*$")

MIN_WORDS, MAX_WORDS = 4, 150


def _clean(seg: str) -> str:
    """Strip the back-reference page lists STEM styles append to each entry."""
    seg = TRAILING_PAGES.sub("", seg.strip())
    return seg.strip(" .;,–-")


def _split_numeric(blob: str, rx: re.Pattern) -> list[str]:
    # re.split with a capturing group interleaves the markers, so the entry text
    # sits at every second element from index 2.
    parts = rx.split(blob)
    return parts[2::2] if len(parts) > 2 else []


def segment(blob: str) -> list[str]:
    """Split a reference blob into entries, choosing the style that fits.

    A style is accepted only if it produces enough entries to be credible for
    the length of the section; otherwise the next style is tried. This stops a
    stray "(3)" in prose from shattering an author-date list into fragments.
    """
    blob = re.sub(r"\s+", " ", blob or "").strip()
    if not blob:
        return []
    words = len(blob.split())
    # A reference averages ~28 words (measured). Requiring a style to reach
    # words//60 rejected real lists that share the section with deposit
    # boilerplate, so every style is scored and the best one wins against a
    # deliberately modest floor.
    floor = max(3, words // 200)

    best = []
    for rx in (BRACKET, PARENNUM, BARENUM):
        segs = [_clean(x) for x in _split_numeric(blob, rx)]
        segs = [x for x in segs if MIN_WORDS <= len(x.split()) <= MAX_WORDS]
        if len(segs) > len(best):
            best = segs
    for rx in (BULLET, AUTHDATE):
        segs = [_clean(x) for x in rx.split(blob)]
        segs = [x for x in segs if MIN_WORDS <= len(x.split()) <= MAX_WORDS]
        if len(segs) > len(best):
            best = segs
    if len(best) >= floor:
        return best

    # Nothing segmented it: hand back the whole blob only if it is short enough
    # to plausibly be one entry, rather than inventing boundaries.
    one = _clean(blob)
    return [one] if MIN_WORDS <= len(one.split()) <= MAX_WORDS else []


def looks_bibliographic(blob: str, segs: list[str] | None = None) -> tuple[bool, str]:
    """Is this section a reference list at all?

    Returns (verdict, reason). Separates 'the parser failed' from 'there was
    nothing here a citation graph could hold' — different numbers that must
    never be added together.

    The decisive evidence is structural: a section that splits into many
    citation-shaped entries IS a reference list, whatever its digit density.
    Judging on density alone rejected ACS and Vancouver lists outright, because
    those styles are numeric by nature ("Nature, 1997. 386 (6626): p. 671-674").
    Density is consulted only when segmentation finds nothing.
    """
    blob = re.sub(r"\s+", " ", blob or "").strip()
    words = blob.split()
    if len(words) < 40:
        return False, "too short"
    if len(YEAR.findall(blob)) < len(words) / 400:
        return False, "almost no years — archival or non-bibliographic list"
    stops = blob.count(".") + blob.count(";")
    if stops and len(words) / stops > 45:
        return False, "long sentences, sparse punctuation — narrative prose, not a list"

    if segs is None:
        segs = segment(blob)
    citation_like = sum(
        1 for s in segs
        if YEAR.search(s) and sum(1 for w in s.split() if w[:1].isupper()) >= 2)
    if len(segs) >= max(3, len(words) // 200) and citation_like >= len(segs) * 0.3:
        return True, "ok"

    numeric_tokens = sum(1 for w in words if w.strip("()[].,;:-").replace(".", "", 1)
                         .replace("-", "").isdigit())
    if numeric_tokens > len(words) * 0.18:
        return False, "numeric-token-dense and unsegmentable — probably a table"
    if sum(1 for w in words if w[:1].isupper()) < len(words) * 0.08:
        return False, "few capitalised tokens — probably not citations"
    return True, "ok"


# --- self-test ---------------------------------------------------------------
# One fixture per citation style actually found in the corpus, plus the two
# kinds of section that are not reference lists at all. Run: refseg.py --selftest
_FIXTURES = [
    ("bracketed numeric",
     "[1] J. D. Lawson. Some criteria for a power producing thermonuclear "
     "reactor. Proceedings of the Physical Society, 6(B70), 1957. "
     "[2] John Wesson. Tokamaks. Clarendon Press, Oxford, third edition, 2004. "
     "[3] W. Horton. Drift waves and transport. Rev. Mod. Phys., 71:735, 1999.", 3),
    ("parenthesised numeric (ACS)",
     "(1) Alberts, B.; Johnson, A.; Lewis, J. Molecular Biology of the Cell; "
     "4th ed.; Garland Science: New York, 2002. "
     "(2) Epping, M. T.; Bernards, R. Int. J. Biochem. Cell Biol. 2009, 41, 16. "
     "(3) Folmer, F.; Orlikova, B. Biochem. Pharmacol. 2010, 80, 1708.", 3),
    ("bulleted (engineering)",
     "- Abbasi, T. and Abbasi, S.A., 2007. Dust explosions - cases, causes and "
     "control. Journal of Hazardous Materials, 140, 7-44. "
     "- AIChE, 1996. Guidelines for use of vapour cloud dispersion models, "
     "second edition, New York. "
     "- Bjerketvedt, D., 1997. Gas explosion handbook. Journal of Hazardous "
     "Materials, 52, 1-150.", 3),
    ("author-date, initial first",
     "Abascal J, Nicolle C (2005). Moving towards inclusive design guidelines "
     "for socially aware HCI. Interacting with Computers, 17: 484-505. "
     "Acuff D, Reiher R (1997). What Kids Buy and Why. The Free Press: New York. "
     "Bell J (2010). Doing Your Research Project. Open University Press.", 3),
    ("author-date after a page range",
     "Abbot, C. (1970) Civic Pride in Chicago. Journal of the Illinois State "
     "Historical Society. 63(4), 399-421 "
     "Ahmed, S. (2004) The Cultural Politics of Emotion. Edinburgh University "
     "Press, Edinburgh, second edition, pages 24-38 "
     "Bhabha, H. (1994) The Location of Culture. Routledge, London, 121-140 "
     "Said, E. (1978) Orientalism. Pantheon Books, New York, pages 1-28 "
     "Spivak, G. (1988) Can the Subaltern Speak? Macmillan, London, 271-313.", 5),
    ("humanities, full first names, transliterated",
     "'Ābādī, Muḥammad Shams al-Ḥaqq, cAwn al-Macbūd, "
     "ed. by cAbd al-Raḥmān (al-Madinah: al-Maktabah al-Salafiyyah 1968). "
     "Abū Dāwūd, Sulaymān, Sunan Abū Dāwūd, "
     "ed. by Muḥammad (Beirut: Dar al-Fikr 1994). "
     "Ibn Ḥajar, Aḥmad, Fatḥ al-Bārī, "
     "ed. by cAbd al-Azīz ibn Bāz (Cairo: al-Matbacah al-Salafiyyah 1959). "
     "al-Bukhārī, Muḥammad ibn Ismācīl, Ṣaḥīḥ al-Bukhārī, "
     "ed. by Muṣṭafā Dīb al-Bughā (Beirut: Dār Ibn Kathīr 1987). "
     "Ibn Taymiyyah, Aḥmad, Majmūc al-Fatāwā, "
     "ed. by cAbd al-Raḥmān al-cĀṣimī (Riyadh: Dār al-cĀṣimah 1995).", 5),
]


def _selftest() -> int:
    bad = 0
    for name, blob, want in _FIXTURES:
        got = segment(blob)
        ok, why = looks_bibliographic(blob)
        if len(got) < want:
            print(f"  FAIL  {name}: segmented {len(got)}, wanted >= {want}")
            for g in got:
                print(f"          | {g[:70]}")
            bad += 1
        elif not ok:
            print(f"  FAIL  {name}: rejected as non-bibliographic ({why})")
            bad += 1
        else:
            print(f"  PASS  {name}: {len(got)} entries")

    # A data table and an archival list must be rejected, not parsed as
    # references — counting them as parse failures overstates recoverable text.
    table = ("Source Lang. n Age SFF SD Range spoken years Hz Hz st "
             "Provonost 1942 US Engl. 6 18 132 122 143 2.8 Mysak 1959 15 32 62 "
             "113 9.5 US Engl. 12 65 79 124 10.0 Hollien 1973 157 129 93 178 "
             "11.3 Lass 1976 30 24 68 121 88 155 9.4 Brown 1991 40 20 90 134 ") * 3
    ok, why = looks_bibliographic(table)
    print(f"  {'PASS' if not ok else 'FAIL'}  data table rejected ({why})")
    bad += int(ok)

    prose = ("One of the Dominican nunneries of Greece was founded in Pera by "
             "the tireless friar William Bernardo of Gaillac who as we saw "
             "reintroduced the Dominicans into Constantinople after the "
             "Byzantine re-conquest and remained there for many years until "
             "the convent was finally dissolved by order of the archbishop ") * 6
    ok, why = looks_bibliographic(prose)
    print(f"  {'PASS' if not ok else 'FAIL'}  narrative prose rejected ({why})")
    bad += int(ok)

    print(f"\n{'REFSEG SELFTEST PASSED' if not bad else f'{bad} FAILURE(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    import sys as _s
    raise SystemExit(_selftest() if "--selftest" in _s.argv else
                     print("usage: refseg.py --selftest") or 0)
