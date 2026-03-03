import pandas as pd
import pytest

from src.phase1_preprocessing.data_loader import DataLoader, LoadSafetyPolicy


def _capture_reports(monkeypatch):
    emitted = []

    def _emit(self, context):
        emitted.append((context, list(self.skip_report)))
        self.reset()

    monkeypatch.setattr(LoadSafetyPolicy, "emit_skip_report", _emit)
    return emitted


def test_smart_load_skips_oversized_file(monkeypatch, tmp_path):
    emitted = _capture_reports(monkeypatch)

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    valid = data_dir / "ok.csv"
    valid.write_text("a,b\n1,2\n")

    oversized = data_dir / "big.csv"
    oversized.write_bytes(b"0" * 12 * 1024 * 1024)  # 12 MB (exceeds min bound 10MB)

    loader = DataLoader(
        str(data_dir),
        skip_on_error=True,
        max_file_size_mb=10,  # At lower bound; 12MB file will be skipped
        max_total_size_mb=50,
        max_memory_mb=50_000,
    )

    df = loader.smart_load(
        pattern="*.csv",
        memory_threshold_mb=10_000,
        chunksize=10_000,
        use_chunking=False,
        io_workers=1,
    )

    assert len(df) == 1
    assert emitted
    found = any(
        context in ("smart_load", "load_csv_files_direct") and
        any(entry["file"] == "big.csv" and "exceeds" in entry["reason"] for entry in skipped)
        for context, skipped in emitted
    )
    assert found


def test_direct_load_skips_parse_error(monkeypatch, tmp_path):
    emitted = _capture_reports(monkeypatch)

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    good = data_dir / "good.csv"
    good.write_text("a,b\n1,2\n")

    bad = data_dir / "bad.csv"
    bad.write_text("x,y\n1,2\n")

    original_read_csv = pd.read_csv

    def fake_read_csv(path, *args, **kwargs):
        if "bad" in str(path):
            raise pd.errors.ParserError("bad csv")
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    loader = DataLoader(
        str(data_dir),
        skip_on_error=True,
        max_file_size_mb=10,
        max_total_size_mb=50,
        max_memory_mb=50_000,
    )

    df = loader.load_csv_files_direct([good, bad])

    assert len(df) == 1
    context, skipped = emitted[-1]
    assert context == "load_csv_files_direct"
    assert any(entry["file"] == "bad.csv" and "parse error" in entry["reason"] for entry in skipped)


def test_direct_load_skips_outside_directory(monkeypatch, tmp_path):
    emitted = _capture_reports(monkeypatch)

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    inside = data_dir / "inside.csv"
    inside.write_text("a,b\n1,2\n")

    outside_dir = tmp_path / "other"
    outside_dir.mkdir()
    outside = outside_dir / "outside.csv"
    outside.write_text("a,b\n3,4\n")

    loader = DataLoader(
        str(data_dir),
        skip_on_error=True,
        max_file_size_mb=10,
        max_total_size_mb=50,
        max_memory_mb=50_000,
    )

    df = loader.load_csv_files_direct([inside, outside])

    assert len(df) == 1
    context, skipped = emitted[-1]
    assert context == "load_csv_files_direct"
    assert any(entry["file"] == "outside.csv" and "outside data directory" in entry["reason"] for entry in skipped)
