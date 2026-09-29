"""Name/address normalization.

Order matters (Implementation_Plan.md stage 02): transliterate non-Latin
scripts *before* stripping accent marks, because Devanagari vowel signs are
combining characters that a premature NFKD accent-strip would mangle. Then
NFKC, casefold, and strip accents. Only after that do we clean noise and
split out legal form / address numbers / postcode.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable, Optional

import anyascii as _anyascii

NOISE_PREFIX_RE = re.compile(r"^\s*(--+|\*\*\*+|>>+|\[\.\.\]|#|@)\s*")
DOMAIN_SUFFIX_RE = re.compile(r"(?<=\w)\.(com|in|fr|net|org|co)\b", re.IGNORECASE)
# Commas/semicolons separate address/name parts but every downstream consumer
# (blocking's word channel, feature Jaccard/cosine, legal-form splitting) only
# ever does whitespace .split() -- so a trailing comma glues two tokens
# together ("road," stays one token, never matching a clean "road"). Turning
# them into spaces first, then dropping periods/brackets/slashes (postal
# abbreviations like "no.", legal forms like "l.l.c." or "pvt.", parenthesized
# words like "(india)"), was measured to affect 40% of address tokens and 12%
# of name tokens.
PUNCT_TO_SPACE_RE = re.compile(r"[,;]")
PUNCT_TO_DROP_RE = re.compile(r"[.()\[\]/]")

DEFAULT_LEGAL_FORMS = {
    "llc", "inc", "corp", "corporation", "co", "ltd", "limited",
    "pvt", "private", "sarl", "sas", "sci", "gmbh", "plc",
}

POSTCODE_TOKEN_RE = re.compile(r"^\d{5,6}$")
ADDR_NUM_TOKEN_RE = re.compile(r"^\d+[a-zA-Z]?$")
VOWELS = set("aeiou")


def is_latin_token(token: str) -> bool:
    """True if the token has no non-Latin alphabetic character (digits,
    punctuation and already-Latin letters all count as Latin)."""
    for c in token:
        if c.isalpha():
            try:
                name = unicodedata.name(c)
            except ValueError:
                continue
            if not name.startswith("LATIN"):
                return False
    return True


def anyascii(token: str) -> str:
    return _anyascii.anyascii(token)


def transliterate_text(
    text: str, translit_dict: dict[str, str], fallback: Callable[[str], str] = anyascii,
) -> tuple[str, bool]:
    """Replace non-Latin tokens using `translit_dict`, falling back to
    `fallback` (anyascii) for tokens the dictionary doesn't cover."""
    tokens = text.split()
    out = []
    changed = False
    for tok in tokens:
        if is_latin_token(tok):
            out.append(tok)
            continue
        changed = True
        hit = translit_dict.get(tok.lower())
        out.append(hit if hit is not None else fallback(tok))
    return " ".join(out), changed


def strip_accents(s: str) -> str:
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def nfkc_casefold(s: str) -> str:
    """NFKC + casefold only, with no accent-stripping and no transliteration.

    Safe to apply to non-Latin text before transliteration, unlike
    `basic_normalize` (whose accent-stripping step would mangle Devanagari
    combining vowel signs). Used to tokenize both sides of a training pair
    when building the Indic->Latin dictionary in dictionaries.py.
    """
    return unicodedata.normalize("NFKC", s).casefold()


def basic_normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = strip_accents(s)
    return s


def clean_noise(s: str) -> str:
    s = NOISE_PREFIX_RE.sub("", s)
    s = DOMAIN_SUFFIX_RE.sub("", s)  # must run before punctuation is dropped, on the "." it matches
    s = PUNCT_TO_SPACE_RE.sub(" ", s)
    s = PUNCT_TO_DROP_RE.sub("", s)
    return " ".join(s.split())


def apply_abbrev(tokens: list[str], country: str, abbrev_map: dict[tuple[str, str], str]) -> list[str]:
    return [abbrev_map.get((country, tok), tok) for tok in tokens]


def consonant_skeleton(s: str) -> str:
    return "".join(c for c in s if c.isalpha() and c not in VOWELS)


def split_legal_form(tokens: list[str], legal_forms: set[str]) -> tuple[list[str], Optional[str]]:
    """Strip a trailing legal-form token (checking a 2-token form first, e.g.
    'private limited') from the end of `tokens`."""
    if len(tokens) >= 2:
        last2 = " ".join(tokens[-2:])
        if last2 in legal_forms:
            return tokens[:-2], last2
    if tokens and tokens[-1] in legal_forms:
        return tokens[:-1], tokens[-1]
    return tokens, None


def normalize_name(
    name: str,
    country: str,
    translit_dict: dict[str, str],
    legal_forms: set[str],
    abbrev_map: Optional[dict[tuple[str, str], str]] = None,
) -> dict:
    text, translit = transliterate_text(name, translit_dict)
    text = basic_normalize(text)
    text = clean_noise(text)
    tokens = text.split()
    if abbrev_map:
        tokens = apply_abbrev(tokens, country, abbrev_map)
    tokens, legal_form = split_legal_form(tokens, legal_forms)
    clean_name = " ".join(tokens)
    nospace = clean_name.replace(" ", "")
    return {
        "name_clean": clean_name,
        "legal_form": legal_form,
        "transliterated": translit,
        "name_nospace": nospace,
        "name_skeleton": consonant_skeleton(nospace),
    }


def extract_postcode_and_numbers(address: str) -> tuple[Optional[str], list[str]]:
    """A leading token is never taken as the postcode: addresses are either
    "number street, city, state" or "state, city, street" (never observed
    leading with a bare postcode), so a 5-6 digit first token is overwhelmingly
    a house number, not a ZIP/PIN -- e.g. "10702 Alicante Way, Rancho
    Cordova, CA" has no ZIP in the text at all, and without this check its
    house number "10702" was wrongly captured as the postcode (and lost from
    addr_numbers, where the number-conflict feature needs it)."""
    tokens = [t.strip(".,;:-") for t in re.split(r"[\s,]+", address.strip()) if t.strip(".,;:-")]
    postcode = None
    numbers = []
    for i, tok in enumerate(tokens):
        if postcode is None and i > 0 and POSTCODE_TOKEN_RE.match(tok):
            postcode = tok
            continue
        if ADDR_NUM_TOKEN_RE.match(tok):
            numbers.append(tok)
    return postcode, numbers


def normalize_address(
    address: str,
    country: str,
    translit_dict: dict[str, str],
    abbrev_map: Optional[dict[tuple[str, str], str]] = None,
) -> dict:
    text, translit = transliterate_text(address, translit_dict)
    text = basic_normalize(text)
    text = clean_noise(text)
    if abbrev_map:
        text = " ".join(apply_abbrev(text.split(), country, abbrev_map))
    postcode, numbers = extract_postcode_and_numbers(text)
    return {
        "address_clean": text,
        "postcode": postcode,
        "addr_numbers": numbers,
        "transliterated": translit,
    }
