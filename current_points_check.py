"""Add `period` to every point and file-level counts to every file.

Each file becomes {"reference_point_count", "current_point_count",
"no_current_points" (only if 0 current points), "points": [...]}.
The script overwrites each file in place. It changes no point field
except adding `period`.
"""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data_json")  # Root folder for standalone runs. start.py ignores it.

# era: (reference start, reference end = current start, current end), all UTC.
ERAS = {
    "era1": ("2025-10-31 03:20:19", "2025-11-07 03:20:19", "2025-11-07 09:20:19"),
    "era2": ("2025-10-28 20:21:50", "2025-11-04 20:21:50", "2025-11-05 02:21:50"),
    "era3": ("2025-10-08 11:15:42", "2025-10-15 11:15:42", "2025-10-15 17:15:42"),
}


def parse(ts):
    """Parse a timestamp as a naive UTC datetime."""
    return datetime.fromisoformat(ts.replace("Z", ""))


def label(t, ref_start, ref_end, cur_end):
    """Return the period of a timestamp. The boundary belongs to `current`."""
    if ref_start <= t < ref_end:
        return "reference"
    if ref_end <= t <= cur_end:
        return "current"
    return None  # The timestamp is outside both windows.


def process_file(path, bounds):
    with open(path) as f:
        data = json.load(f)
    # A file from an earlier run is a dict. A fresh file is a list.
    points = data["points"] if isinstance(data, dict) else data
    periods = [label(parse(p["timestamp"]), *bounds) for p in points]
    for p, period in zip(points, periods):
        p["period"] = period
    out = {
        "reference_point_count": periods.count("reference"),
        "current_point_count": periods.count("current"),
    }
    if out["current_point_count"] == 0:
        out["no_current_points"] = True
    out["points"] = points
    # Write to a temp file first so a failure cannot corrupt the original.
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)


def run():
    for era, (rs, re_, ce) in ERAS.items():
        bounds = (parse(rs), parse(re_), parse(ce))
        for path in sorted((DATA_DIR / era).iterdir()):
            if path.suffix == ".json":  # Skip other file types.
                process_file(path, bounds)


if __name__ == "__main__":
    run()