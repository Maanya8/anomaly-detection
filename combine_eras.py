"""Combine every era's per-satellite JSON files into one file.

Usage:
    python combine_eras.py <data_dir> <output_file>

`data_dir` holds era1/, era2/, and era3/, or any era* folders, of
per-satellite JSON files. Each file can be a plain list of points or the
wrapped {reference_point_count, ..., points} structure. The combined file
nests the files by era and original file stem, so `split_eras.py` can
rebuild every file exactly:

    {"era1": {"1": <content of era1/1.json>, "2": <...>, ...}, "era2": {...}}
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def combine(data_dir, output_file):
    """Read every era*/*.json file under data_dir into the nested
    {era: {stem: content}} structure described above.

    If a file fails to parse, skip it and continue, so one bad file does
    not stop the rest. The function returns the skipped files separately
    so the caller can report them.
    """
    data_dir = Path(data_dir)
    combined = {}
    skipped = []
    for era_dir in sorted(data_dir.glob("era*")):
        if not era_dir.is_dir():
            continue
        era_files = {}
        for path in sorted(era_dir.glob("*.json")):
            try:
                with open(path) as f:
                    era_files[path.stem] = json.load(f)
            except Exception as exc:
                skipped.append((str(path), str(exc)))
        combined[era_dir.name] = era_files
    return combined, skipped


def write_combined(combined, output_file):
    # Write to a temp file first so a failure cannot leave a truncated
    # `output_file`.
    output_file = Path(output_file)
    fd, tmp = tempfile.mkstemp(dir=output_file.parent or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(combined, f)
    os.replace(tmp, output_file)


def _check():
    """Check that combining and then splitting a small synthetic era tree
    with `split_eras.split` restores every file's content exactly."""
    import split_eras

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "data_json"
        (src / "era1").mkdir(parents=True)
        (src / "era2").mkdir(parents=True)
        (src / "era1" / "1.json").write_text(json.dumps([{"norad_id": 1, "magnitude": 12.3}]))
        (src / "era1" / "2.json").write_text(json.dumps(
            {"reference_point_count": 0, "current_point_count": 0, "points": []}))
        (src / "era2" / "1.json").write_text(json.dumps([{"norad_id": 1, "magnitude": 9.9}]))

        combined_path = tmp / "combined.json"
        combined, skipped = combine(src, combined_path)
        assert not skipped, skipped
        write_combined(combined, combined_path)

        out_dir = tmp / "restored"
        written, split_skipped = split_eras.split(combined_path, out_dir)
        assert not split_skipped, split_skipped
        assert written == 3

        for rel in ("era1/1.json", "era1/2.json", "era2/1.json"):
            original = json.loads((src / rel).read_text())
            restored = json.loads((out_dir / rel).read_text())
            assert original == restored, f"{rel} did not round-trip"
    print("check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", nargs="?", help="Folder with era1/era2/era3 of per-satellite JSON files")
    parser.add_argument("output_file", nargs="?", help="Path to write the combined JSON file to")
    parser.add_argument("--check", action="store_true", help="Run the round-trip self-test and exit")
    args = parser.parse_args()

    if args.check:
        _check()
        return

    if not args.data_dir or not args.output_file:
        parser.error("data_dir and output_file are required unless --check is given")

    combined, skipped = combine(args.data_dir, args.output_file)
    write_combined(combined, args.output_file)

    n_files = sum(len(v) for v in combined.values())
    print(f"Combined {n_files} file(s) across {len(combined)} era(s) into {args.output_file}")
    for path, error in skipped:
        print(f"  skipped {path}: {error}")
    if skipped:
        sys.exit(1)


if __name__ == "__main__":
    main()