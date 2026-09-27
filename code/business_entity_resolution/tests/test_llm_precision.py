import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.llm import choose_precision

GIB = 2**30


def test_16gb_card_loads_bf16_even_with_bitsandbytes():
    assert choose_precision("cuda", 16 * GIB, bnb_available=True) == "bf16"


def test_8gb_card_uses_nf4():
    # an RTX 4060 laptop reports ~7.6 GiB total
    assert choose_precision("cuda", int(7.6 * GIB), bnb_available=True) == "nf4"


def test_12gb_card_still_uses_nf4():
    assert choose_precision("cuda", 12 * GIB, bnb_available=True) == "nf4"


def test_cpu_is_fp32():
    assert choose_precision("cpu", None, bnb_available=True) == "fp32"
