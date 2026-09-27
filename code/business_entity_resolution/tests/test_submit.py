"""The streaming candidate_pairs.tsv check must flag the same problems the
reference validator does."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.submit import check_candidates


def _write(path: Path, header: str, rows: list[str]) -> Path:
    path.write_text(header + "\n" + "".join(r + "\n" for r in rows))
    return path


def _test_dir(tmp_path: Path) -> Path:
    d = tmp_path / "test"
    d.mkdir()
    _write(d / "test_source1.tsv", "entity_id\tbusiness_name", ["S1-1\ta", "S1-2\tb"])
    _write(d / "test_source2.tsv", "entity_id\tbusiness_name", ["S2-5\tx"])
    _write(d / "test_source3.tsv", "entity_id\tbusiness_name", ["S3-7\ty"])
    return d


def test_clean_file_passes(tmp_path):
    d = _test_dir(tmp_path)
    cand = _write(tmp_path / "c.tsv", "source1_entity_id\tcandidate_entity_ids", ["S1-1\tS2-5,S3-7", "S1-2\t"])
    match = _write(tmp_path / "m.tsv", "source1_entity_id\tmatched_entity_ids", ["S1-1\tS2-5", "S1-2\t"])
    errors, warnings = check_candidates(cand, d, match, check_ids=True)
    assert errors == [] and warnings == []


def test_each_rule_is_flagged(tmp_path):
    d = _test_dir(tmp_path)
    cand = _write(tmp_path / "c.tsv", "source1_entity_id\tcandidate_entity_ids",
                  ["S1-1\tS2-5,S2-5,S1-9,X-3,S2-404", "S1-1\t", "S1-99\t"])
    match = _write(tmp_path / "m.tsv", "source1_entity_id\tmatched_entity_ids", ["S1-1\tS3-7"])
    errors, warnings = check_candidates(cand, d, match, check_ids=True)
    text = " ".join(errors)
    for expected in ("duplicate source1_entity_id", "repeated ID", "S2-/S3- prefix",
                     "not in the test Source-2/3", "missing", "not in the test set"):
        assert expected in text, expected
    assert any("not present" in w for w in warnings)
