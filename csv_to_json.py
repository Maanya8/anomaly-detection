"""Convert every CSV file under a folder to JSON and keep the folder structure.

Usage:
    python csv_to_json.py <input_dir> <output_dir>

Each CSV row becomes one JSON object. Numeric text becomes a number, except in
text columns. Hidden folders, such as `.ipynb_checkpoints`, are skipped.
"""

import csv
import json
import sys
from pathlib import Path

# Keep these columns as text. Some sensor IDs look like numbers, for example
# `62889e77` reads as 6.2889e77, so converting them would corrupt the ID.
TEXT_COLUMNS = {"timestamp", "sensor"}


def to_number(text):
    """Return `text` as an int or a float if possible, else the original text."""
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def convert(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    count = 0
    for csv_path in sorted(input_dir.rglob("*.csv")):
        relative = csv_path.relative_to(input_dir)
        # Skip notebook checkpoint copies. They duplicate real files.
        if any(part.startswith(".") for part in relative.parts):
            continue
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            # Drop the unnamed first column. It is a saved pandas row index, not data.
            rows = [
                {k: v if k in TEXT_COLUMNS else to_number(v) for k, v in row.items() if k}
                for row in reader
            ]
        json_path = output_dir / relative.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows))
        count += 1
    print(f"Converted {count} files from {input_dir} to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python csv_to_json.py <input_dir> <output_dir>")
    convert(sys.argv[1], sys.argv[2])