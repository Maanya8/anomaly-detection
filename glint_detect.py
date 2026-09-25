"""Add a `glint` field to every point in the era JSON files.

The script overwrites each file in place. It changes no other field.

The script accepts two file shapes:
  - a plain list of points, for example [ {...}, {...} ]
  - the wrapped structure that `current_points_check.py` writes, for example
    {"reference_point_count": N, "current_point_count": M,
     "no_current_points": true,  "points": [ {...}, {...} ]}
The script writes back the same shape it reads. It sets only the `glint`
field of each point and keeps every other field, including the wrapper
counts.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

DATA_DIR = Path("data_json")  # Root folder for standalone runs. start.py ignores it.
ERAS = ["era1", "era2", "era3"]

FIXED_WINDOW = (-5.0, 10.0)  # fallback window in degrees of SEPA
ZONE = 20.0                  # search for glint where |SEPA| <= ZONE
BASE_MAX = 40.0              # baseline bins: ZONE < |SEPA| <= BASE_MAX
BIN = 1.0                    # bin width in degrees
MIN_POINTS = 30              # minimum points in the zone for the data-driven window
MIN_BIN_POINTS = 3           # a bin needs this many points to get a spread value
SPREAD_FACTOR = 3.0          # a bin is glint if its spread exceeds this times baseline


def bin_spreads(sepa, mag):
    """Return {bin_start: max(mag) - min(mag)} for bins with enough points."""
    starts = np.floor(sepa / BIN) * BIN
    spreads = {}
    for b in np.unique(starts):
        m = mag[starts == b]
        if len(m) >= MIN_BIN_POINTS:
            spreads[b] = m.max() - m.min()
    return spreads


def glint_window(sepa, mag):
    """Return (low, high) SEPA bounds of the glint, or None if no glint is found.

    Glint scatters the magnitude widely inside a narrow SEPA range. The window
    is the range of zone bins whose magnitude spread is much larger than the
    spread in the baseline bins farther from 0 degrees.
    """
    a = np.abs(sepa)
    zone = a <= ZONE
    base = (a > ZONE) & (a <= BASE_MAX)
    if zone.sum() < MIN_POINTS:
        return FIXED_WINDOW
    base_spreads = bin_spreads(sepa[base], mag[base])
    if not base_spreads:
        return FIXED_WINDOW  # no baseline to compare against
    threshold = SPREAD_FACTOR * np.median(list(base_spreads.values()))
    flagged = [b for b, s in bin_spreads(sepa[zone], mag[zone]).items() if s > threshold]
    if not flagged:
        return None
    return min(flagged), max(flagged) + BIN


def load_points(data):
    """Return (points, wrapper).

    If the file is wrapped, `wrapper` is the original dict, and `points` is
    the same list object as `wrapper["points"]`. If the file is a plain
    list, `wrapper` is None.
    """
    if isinstance(data, dict) and "points" in data:
        return data["points"], data
    return data, None


def process_file(path):
    with open(path) as f:
        data = json.load(f)

    points, wrapper = load_points(data)

    if points:
        sepa = np.array([p["equatorial_phase"] for p in points], dtype=float)
        mag = np.array([p["magnitude"] for p in points], dtype=float)
        window = glint_window(sepa, mag)
    else:
        # The file has no points, so no point is glint.
        sepa = np.array([], dtype=float)
        window = None

    count = 0
    for p, s in zip(points, sepa):
        p["glint"] = bool(window is not None and window[0] <= s <= window[1])
        if(p["glint"] == True):
          count += 1

    print(f"{path}:{count} glint points")

    # For a wrapped file, `points` is the same list object as
    # `wrapper["points"]`, so `wrapper` already holds the new glint values.
    # Write back the shape that was read.
    out = wrapper if wrapper is not None else points

    # Write to a temp file first so a failure cannot corrupt the original.
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)


def diagnose_file(path):
    """Report which branch `glint_window()` takes for a file, and why.

    The function writes nothing. Use it to find out why a file has no glint
    points, or more or fewer than expected.
    """
    with open(path) as f:
        data = json.load(f)
    points, _ = load_points(data)

    info = {"file": path.name, "n_points": len(points)}
    if not points:
        info["mode"] = "empty (no points)"
        return info

    sepa = np.array([p["equatorial_phase"] for p in points], dtype=float)
    mag = np.array([p["magnitude"] for p in points], dtype=float)

    info["sepa_min"] = float(np.nanmin(sepa))
    info["sepa_max"] = float(np.nanmax(sepa))
    info["n_nan_sepa"] = int(np.isnan(sepa).sum())
    info["n_nan_mag"] = int(np.isnan(mag).sum())

    a = np.abs(sepa)
    zone = a <= ZONE
    base = (a > ZONE) & (a <= BASE_MAX)
    info["zone_n"] = int(zone.sum())
    info["base_n"] = int(base.sum())

    if zone.sum() < MIN_POINTS:
        window = FIXED_WINDOW
        info["mode"] = f"fixed fallback (zone_n={zone.sum()} < MIN_POINTS={MIN_POINTS})"
    else:
        base_spreads = bin_spreads(sepa[base], mag[base])
        if not base_spreads:
            window = FIXED_WINDOW
            info["mode"] = "fixed fallback (no baseline bins with >= MIN_BIN_POINTS)"
        else:
            zone_spreads = bin_spreads(sepa[zone], mag[zone])
            threshold = SPREAD_FACTOR * np.median(list(base_spreads.values()))
            flagged = [b for b, s in zone_spreads.items() if s > threshold]
            info["baseline_median_spread"] = float(np.median(list(base_spreads.values())))
            info["threshold"] = float(threshold)
            info["zone_bins_with_data"] = len(zone_spreads)
            info["max_zone_bin_spread"] = float(max(zone_spreads.values())) if zone_spreads else None
            if flagged:
                window = (min(flagged), max(flagged) + BIN)
                info["mode"] = "data-driven (glint found)"
            else:
                window = None
                info["mode"] = "data-driven (no bin exceeded threshold -> no glint)"

    info["window"] = window
    if window is not None:
        in_window = int(((sepa >= window[0]) & (sepa <= window[1])).sum())
        info["points_in_window"] = in_window

    return info


def diagnose():
    for era in ERAS:
        era_dir = DATA_DIR / era
        if not era_dir.exists():
            continue
        for path in sorted(era_dir.iterdir()):
            if path.suffix != ".json":
                continue
            info = diagnose_file(path)
            print(f"\n{era}/{info['file']}  (n_points={info.get('n_points')})")
            for k, v in info.items():
                if k in ("file", "n_points"):
                    continue
                print(f"  {k}: {v}")


def run():
    for era in ERAS:
        for path in sorted((DATA_DIR / era).iterdir()):
            if path.suffix == ".json":  # Skip other file types.
                process_file(path)


def _check():
    """Check that a synthetic satellite with glint scatter at 3 to 8 degrees
    gets a window near those bounds."""
    rng = np.random.default_rng(0)
    s = rng.uniform(-40, 40, 3000)
    m = 12 + 0.02 * s + rng.normal(0, 0.05, s.size)
    g = (s >= 3) & (s <= 8)
    m[g] += rng.uniform(-2, 2, g.sum())
    lo, hi = glint_window(s, m)
    assert 2 <= lo <= 4 and 8 <= hi <= 9, (lo, hi)
    print("check passed:", lo, hi)


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
    elif "--diagnose" in sys.argv:
        diagnose()
    else:
        run()