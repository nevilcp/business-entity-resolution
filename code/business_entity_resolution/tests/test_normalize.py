import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.normalize import (
    basic_normalize,
    clean_noise,
    is_latin_token,
    normalize_address,
    normalize_name,
    transliterate_text,
)


def test_devanagari_falls_back_to_anyascii_when_dict_misses():
    text = "राम मार्केटिंग प्राइवेट लिमिटेड"
    out, changed = transliterate_text(text, translit_dict={})
    assert changed is True
    assert is_latin_token(out)
    assert out.strip() != ""


def test_devanagari_prefers_the_dictionary_over_anyascii():
    text = "राम मार्केटिंग"
    out, changed = transliterate_text(text, translit_dict={"राम": "ram", "मार्केटिंग": "marketing"})
    assert out == "ram marketing"
    assert changed is True


def test_noise_prefix_is_stripped():
    assert clean_noise("-- Holloway Peak Inc Seafood") == "Holloway Peak Inc Seafood"
    assert clean_noise("*** Acme Corp") == "Acme Corp"
    assert clean_noise(">> Acme Corp") == "Acme Corp"


def test_domain_suffix_is_stripped():
    assert clean_noise("capitalholding.com") == "capitalholding"
    assert clean_noise("boulangerie.fr") == "boulangerie"


def test_french_accents_are_folded_and_casefolded():
    assert basic_normalize("Café Éclair") == "cafe eclair"
    assert basic_normalize("Boulangerie Pâtisserie") == "boulangerie patisserie"


def test_normalize_name_splits_legal_form_and_flags_transliteration():
    result = normalize_name(
        "-- Acme Private Limited", country="India",
        translit_dict={}, legal_forms={"private limited"},
    )
    assert result["name_clean"] == "acme"
    assert result["legal_form"] == "private limited"
    assert result["transliterated"] is False


def test_normalize_name_transliterates_and_skeletonizes():
    result = normalize_name(
        "राम मार्केटिंग", country="India",
        translit_dict={"राम": "ram", "मार्केटिंग": "marketing"}, legal_forms=set(),
    )
    assert result["name_clean"] == "ram marketing"
    assert result["name_nospace"] == "rammarketing"
    assert result["name_skeleton"] == "rmmrktng"
    assert result["transliterated"] is True


def test_normalize_address_extracts_postcode_and_numbers():
    result = normalize_address("221 Baker Street, London, 411045", country="India", translit_dict={})
    assert result["postcode"] == "411045"
    assert "221" in result["addr_numbers"]
