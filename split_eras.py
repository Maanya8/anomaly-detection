"""Split one combined JSON file back into per-era, per-satellite files.

Usage:
    python split_eras.py <combined_file> <output_dir>

Reverses combine_eras.py: combined_file is the nested
{era: {stem: content}} structure that script writes. output_dir is
created if it doesn't exist and reused as-is if it does; every
<era>/<stem>.json is written under it with exactly the content
combine_eras.py read from the original file.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def split(combined_file, output_dir):
    """Read combined_file and write every era/stem.json under output_dir.

    A file that fails to write is skipped, not fatal, for the same reason
    combine_eras.py skips a file that fails to parse: one bad entry should
    not block writing the rest. Skipped entries come back separately so
    the caller can report them.
    """
    with open(combined_file) as f:
        combined = json.load(f)

    output_dir = Path(output_dir)
    written = 0
    skipped = []
    for era, files in combined.items():
        era_dir = output_dir / era
        era_dir.mkdir(parents=True, exist_ok=True)
        for stem, content in files.items():
            path = era_dir / f"{stem}.json"
            try:
                _write_json(path, content)
                written += 1
            except Exception as exc:
                skipped.append((str(path), str(exc)))
    return written, skipped


def _write_json(path, content):
    # Write to a temp file first so a failure cannot leave a truncated
    # file behind, same pattern the rest of the pipeline uses.
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(content, f)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("combined_file", nargs="?", help="Combined JSON file written by combine_eras.py")
    parser.add_argument("output_dir", nargs="?", help="Folder to write era1/era2/era3 subfolders into")
    parser.add_argument("--check", action="store_true", help="Run the round-trip self-test and exit")
    args = parser.parse_args()

    if args.check:
        import combine_eras
        combine_eras._check()
        return

    if not args.combined_file or not args.output_dir:
        parser.error("combined_file and output_dir are required unless --check is given")

    written, skipped = split(args.combined_file, args.output_dir)
    print(f"Wrote {written} file(s) to {args.output_dir}")
    for path, error in skipped:
        print(f"  skipped {path}: {error}")
    if skipped:
        sys.exit(1)


if __name__ == "__main__":
    main()