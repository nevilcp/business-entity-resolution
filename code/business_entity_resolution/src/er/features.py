"""Pairwise name/address/graph features for the GBDT scorer (stage 06).

Name/address channel scores here are recomputed directly from the
normalized text at pair grain (rather than threaded through from stage 04's
per-candidate blocking scores), since candidate_pairs.tsv is already the
pruned set and this keeps the feature step self-contained. The "graph"
features' reverse-rank/gap come from stage 04's per-record aggregates,
which really do need the full blocking pass to compute (see blocking.py).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .blocking import char_ngrams


def token_jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def idf_cosine(tokens_a: list[str], tokens_b: list[str], idf: dict[str, float]) -> float:
    wa = {t: idf.get(t, 0.0) for t in set(tokens_a)}
    wb = {t: idf.get(t, 0.0) for t in set(tokens_b)}
    common = set(wa) & set(wb)
    num = sum(wa[t] * wb[t] for t in common)
    na = np.sqrt(sum(v * v for v in wa.values()))
    nb = np.sqrt(sum(v * v for v in wb.values()))
    return num / (na * nb) if na > 0 and nb > 0 else 0.0


def ngram_cosine(a: str, b: str, n: int = 3) -> float:
    ga, gb = set(char_ngrams(a, n)), set(char_ngrams(b, n))
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / np.sqrt(len(ga) * len(gb))


def acronym(tokens: list[str]) -> str:
    return "".join(t[0] for t in tokens if t)


def acronym_match(tokens_a: list[str], tokens_b: list[str], nospace_a: str, nospace_b: str) -> float:
    return float(acronym(tokens_a) == nospace_b or acronym(tokens_b) == nospace_a)


def abbrev_subseq_match(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Fraction of the shorter name's tokens that are a prefix of some token
    in the longer name (catches 'intl' vs 'international')."""
    short, long_ = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    if not short:
        return 0.0
    hits = sum(1 for t in short if any(lt.startswith(t) for lt in long_))
    return hits / len(short)


def legal_form_feature(lf_a: Optional[str], lf_b: Optional[str]) -> int:
    """2 = equal, 1 = different, 0 = at least one missing."""
    if lf_a is None or lf_b is None:
        return 0
    return 2 if lf_a == lf_b else 1


def name_features(rec_a: dict, rec_b: dict, idf: dict[str, float]) -> dict:
    tok_a, tok_b = rec_a["name_clean"].split(), rec_b["name_clean"].split()
    return {
        "name_jaccard": token_jaccard(set(tok_a), set(tok_b)),
        "name_idf_cosine": idf_cosine(tok_a, tok_b, idf),
        "name_ngram_cosine": ngram_cosine(rec_a["name_nospace"], rec_b["name_nospace"]),
        "name_ratio": fuzz.ratio(rec_a["name_clean"], rec_b["name_clean"]) / 100.0,
        "name_token_sort": fuzz.token_sort_ratio(rec_a["name_clean"], rec_b["name_clean"]) / 100.0,
        "name_token_set": fuzz.token_set_ratio(rec_a["name_clean"], rec_b["name_clean"]) / 100.0,
        "name_jaro_winkler": JaroWinkler.normalized_similarity(rec_a["name_clean"], rec_b["name_clean"]),
        "name_abbrev_subseq": abbrev_subseq_match(tok_a, tok_b),
        "name_acronym_match": acronym_match(tok_a, tok_b, rec_a["name_nospace"], rec_b["name_nospace"]),
        "legal_form": legal_form_feature(rec_a.get("legal_form"), rec_b.get("legal_form")),
        "name_transliterated": float(bool(rec_a.get("name_transliterated")) or bool(rec_b.get("name_transliterated"))),
    }


def address_features(rec_a: dict, rec_b: dict, idf: dict[str, float]) -> dict:
    a, b = rec_a["address_clean"], rec_b["address_clean"]
    tok_a, tok_b = a.split(), b.split()
    nums_a, nums_b = set(rec_a.get("addr_numbers") or []), set(rec_b.get("addr_numbers") or [])
    shared_nums = nums_a & nums_b
    conflict = bool(nums_a and nums_b and not shared_nums)
    pc_a, pc_b = rec_a.get("postcode"), rec_b.get("postcode")
    return {
        "addr_jaccard": token_jaccard(set(tok_a), set(tok_b)),
        "addr_idf_cosine": idf_cosine(tok_a, tok_b, idf),
        "addr_token_set": fuzz.token_set_ratio(a, b) / 100.0 if a and b else 0.0,
        "addr_shared_numbers": len(shared_nums),
        "addr_number_conflict": float(conflict),
        "addr_postcode_equal": float(bool(pc_a) and pc_a == pc_b),
        "addr_missing": float(not a or not b),
    }


def build_idf(token_lists: list[list[str]]) -> dict[str, float]:
    n = len(token_lists)
    df: dict[str, int] = {}
    for toks in token_lists:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    return {t: float(np.log(n / (1 + c))) for t, c in df.items()}


def build_idf_chunked(chunks) -> dict[str, float]:
    """`build_idf` over an iterable of token-list chunks, so a multi-million
    record pool never has to exist as one list of token lists."""
    from collections import Counter
    from itertools import chain

    df: Counter = Counter()
    n = 0
    for token_lists in chunks:
        n += len(token_lists)
        df.update(chain.from_iterable(map(set, token_lists)))
    return {t: float(np.log(n / (1 + c))) for t, c in df.items()}


_DUMMY = {"name_clean": "a", "name_nospace": "a", "legal_form": None, "name_transliterated": False,
          "address_clean": "a", "postcode": None, "addr_numbers": []}
PAIR_FEATURE_NAMES = list(name_features(_DUMMY, _DUMMY, {})) + list(address_features(_DUMMY, _DUMMY, {}))


def pair_feature_matrix(recs_a: list[dict], recs_b: list[dict], idf: dict[str, float]) -> np.ndarray:
    """(n_pairs x len(PAIR_FEATURE_NAMES)) float32 of name + address
    features for aligned record lists."""
    out = np.empty((len(recs_a), len(PAIR_FEATURE_NAMES)), dtype=np.float32)
    for i, (a, b) in enumerate(zip(recs_a, recs_b)):
        nf = name_features(a, b, idf)
        af = address_features(a, b, idf)
        out[i] = list(nf.values()) + list(af.values())
    return out
