"""Per-country/per-pool dictionaries used (not recomputed) by normalize.py:
legal-form vocabulary, the Indic->Latin transliteration map, and the
abbreviation expansion map.

These are unsupervised frequency statistics over a *pool* of names (like the
per-pool IDF stage 04 computes for blocking), not model parameters, so
they're computed separately for the train pool and the test pool -- the
Indic dictionary is the one exception, since it needs training India
ground-truth pairs to positionally align tokens (see build_indic_dict) and
is then reused as-is to normalize test India names.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Iterable, Optional

from .normalize import DEFAULT_LEGAL_FORMS, is_latin_token


def build_legal_form_vocab(
    names_by_country: dict[str, Iterable[str]],
    seed_forms: set[str] = DEFAULT_LEGAL_FORMS,
    min_count: int = 50,
    top_n: int = 30,
) -> dict[str, set[str]]:
    """Seed legal forms plus each country's most frequent name-final tokens,
    so a country with no seed coverage (or no training presence, like
    France) is still covered as long as it appears in the pool being
    normalized.
    """
    vocab = {}
    for country, names in names_by_country.items():
        counts = Counter()
        for name in names:
            toks = name.split()
            if toks:
                counts[toks[-1]] += 1
        frequent = {t for t, c in counts.most_common(top_n) if c >= min_count}
        vocab[country] = set(seed_forms) | frequent
    return vocab


def positional_align_indic_dict(
    pairs: Iterable[tuple[list[str], list[str]]], min_support: int = 2,
) -> dict[str, str]:
    """`pairs` is (non_latin_name_tokens, latin_name_tokens) for training
    India (S1, match) pairs. Aligns same-position tokens and keeps the
    mapping for a source token whose most common aligned target appears at
    least `min_support` times.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for src_tokens, dst_tokens in pairs:
        for s, d in zip(src_tokens, dst_tokens):
            if not is_latin_token(s) and is_latin_token(d):
                counts[s.lower()][d.lower()] += 1
    out = {}
    for tok, c in counts.items():
        target, support = c.most_common(1)[0]
        if support >= min_support:
            out[tok] = target
    return out


def extend_dict_with_llm(
    base_dict: dict[str, str],
    missing_tokens: Iterable[str],
    llm_transliterate: Callable[[list[str]], list[str]],
) -> dict[str, str]:
    """Fill in tokens `positional_align_indic_dict` didn't cover using a
    batched, greedy-decoding LLM call (injected as `llm_transliterate` so
    this module never has to import a model itself).
    """
    tokens = sorted({t.lower() for t in missing_tokens if t.lower() not in base_dict})
    if not tokens:
        return base_dict
    latin = llm_transliterate(tokens)
    out = dict(base_dict)
    for tok, latin_tok in zip(tokens, latin):
        if latin_tok:
            out[tok] = latin_tok.lower()
    return out


def frequent_short_tokens(
    names_by_country: dict[str, Iterable[str]], max_len: int = 3, top_n: int = 50, min_count: int = 20,
) -> dict[str, list[str]]:
    out = {}
    for country, names in names_by_country.items():
        counts = Counter()
        for name in names:
            for tok in name.split():
                if len(tok) <= max_len:
                    counts[tok] += 1
        out[country] = [t for t, c in counts.most_common(top_n) if c >= min_count]
    return out


def build_abbrev_map(
    names_by_country: dict[str, Iterable[str]],
    llm_expand: Callable[[list[str], str], list[Optional[str]]],
    max_len: int = 3,
    top_n: int = 50,
    min_count: int = 20,
) -> dict[tuple[str, str], str]:
    """Sends frequent short tokens per country to the LLM (with the country
    name in the prompt) and accepts an expansion only if it already exists
    in that country's own vocabulary -- the hallucination guard from the
    plan.
    """
    candidates = frequent_short_tokens(names_by_country, max_len, top_n, min_count)
    vocab_by_country = {
        country: {w for name in names for w in name.split() if len(w) > max_len}
        for country, names in names_by_country.items()
    }
    abbrev_map: dict[tuple[str, str], str] = {}
    for country, tokens in candidates.items():
        if not tokens:
            continue
        expansions = llm_expand(tokens, country)
        vocab = vocab_by_country.get(country, set())
        for tok, exp in zip(tokens, expansions):
            if exp and exp.lower() in vocab:
                abbrev_map[(country, tok)] = exp.lower()
    return abbrev_map
