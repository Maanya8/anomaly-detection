"""Score current-window points against reference points in a fixed SEPA window.

The scoring matches `knn.py`, but the neighbors differ. `knn.py` takes the k
nearest reference points by SEPA. This script takes every reference point
inside a fixed SEPA window around the current point, which is a rolling
median.

    spread = sqrt( (1.4826 * MAD(neighbor magnitudes))^2 + magnitude_unc^2 )
    z_score_median = (current magnitude - median(neighbor magnitudes)) / spread

The script only adds fields. It keeps every field that earlier stages wrote,
including the `knn.py` fields. It adds these fields:

    per point (current, non-glint, with enough neighbors in the window):
        "z_score_median"       -> float
        "flagged_median"       -> bool
        "anomaly_score_median" -> float, only if the point is in an anomaly
    at the file level:
        "anomalies_median" -> [ {norad_id, equatorial_phase, timestamp,
                                   score, n_points}, ... ]

Like `knn.py`, the script ignores glint points. They are never neighbors and
never get a score.

The script overwrites each JSON file in place through a temp file and
`os.replace`. It saves one diagnostic PNG per file to `PLOTS_DIR`. Each PNG
name ends in "_median" so it cannot overwrite a `knn.py` plot.
"""

import glob
import json
import math
import os
import tempfile
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Editable parameters. Change these values and re-run the script.
# ---------------------------------------------------------------------------

# Root folder that holds the era1/, era2/, and era3/ subfolders of
# per-satellite JSON files. `start.py` replaces this value with its own
# `data_dir`.
DATA_DIR = "data_json"

# Folder for the diagnostic plots. The script writes one PNG per satellite
# file and repeats the era1/era2/era3 subfolder layout. This folder can be
# the same one `knn.py` uses, because the "_median" suffix keeps names apart.
PLOTS_DIR = "plots_median"

# Half-width of the rolling window, in degrees of SEPA. A current point at
# SEPA s uses every non-glint reference point with SEPA in [s - W, s + W].
#   Narrower window: more specific to the local SEPA, but sparse regions can
#                    have too few reference points.
#   Wider window:    a more stable median and MAD, but it can hide local
#                    structure and include reference points from a
#                    different SEPA.
# No standard value exists. Try 2.0, 5.0, and 10.0 degrees and compare the
# results with `knn.py`.
WINDOW_HALF_WIDTH_DEG = 5.0

# Minimum number of reference points the window needs before the script
# computes a score. With fewer points the local median and MAD are
# unreliable, so the point gets no `z_score_median` field.
MIN_NEIGHBORS = 5

# A point with |z_score_median| at or above this value is flagged and can
# join an anomaly. The tested range from `knn.py` (2.5, 3.0, 3.5) also applies
# here. At the same point, the window can select different neighbors than
# kNN, so the z-scores can differ.
FLAG_THRESHOLD = 3.0

# ---------------------------------------------------------------------------


def robust_mad(values, center):
    """Return the median absolute deviation around `center`, unscaled by 1.4826."""
    return statistics_median(abs(v - center) for v in values)


def statistics_median(iterable):
    vals = sorted(iterable)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def window_neighbors(target_sepa, ref_points, half_width):
    """Return every reference point with SEPA in [target_sepa - half_width,
    target_sepa + half_width]."""
    lo, hi = target_sepa - half_width, target_sepa + half_width
    return [p for p in ref_points if lo <= p["equatorial_phase"] <= hi]


def point_z_score_median(current_point, ref_points, half_width, min_neighbors):
    """
    Return the rolling-median robust z-score of one current point. The
    neighbors are every reference point inside a fixed SEPA window. The
    point's own `magnitude_unc` is added in quadrature. If the window holds
    fewer than `min_neighbors` reference points, return None.
    """
    neighbors = window_neighbors(current_point["equatorial_phase"], ref_points, half_width)
    if len(neighbors) < min_neighbors:
        return None

    mags = [p["magnitude"] for p in neighbors]
    median_mag = statistics_median(mags)
    mad = robust_mad(mags, median_mag)
    ref_scatter = 1.4826 * mad
    point_unc = current_point.get("magnitude_unc", 0.0) or 0.0

    spread = math.sqrt(ref_scatter ** 2 + point_unc ** 2)
    if spread == 0:
        # Both the reference scatter and the reported uncertainty are zero.
        # Use a small epsilon to avoid a ZeroDivisionError. An exact match
        # still gives z == 0.
        spread = 1e-6

    return (current_point["magnitude"] - median_mag) / spread


def group_and_score_anomalies_median(scored_current_points, threshold):
    """
    Group consecutive flagged points into anomalies, as `knn.py` does.

    Points are in time order, and a group ends at the first unflagged point.
    The normal CDF turns the median |z_score_median| of each group into its
    score.
    """
    anomalies = []
    point_to_score = {}
    group = []

    def close_group():
        if not group:
            return
        abs_zs = [abs(p["z_score_median"]) for p in group]
        median_abs_z = statistics_median(abs_zs)
        score = math.erf(median_abs_z / math.sqrt(2))
        sepas = [p["equatorial_phase"] for p in group]
        times = [p["timestamp"] for p in group]
        anomalies.append({
            "norad_id": group[0]["norad_id"],
            "equatorial_phase": [min(sepas), max(sepas)],
            "timestamp": [min(times), max(times)],
            "score": score,
            "n_points": len(group),
        })
        for p in group:
            point_to_score[id(p)] = score

    for p in scored_current_points:
        p["flagged_median"] = abs(p["z_score_median"]) >= threshold
        if p["flagged_median"]:
            group.append(p)
        else:
            close_group()
            group = []
    close_group()  # Close the last group if the data ends on a flagged point.

    return anomalies, point_to_score


def parse_timestamp(ts):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return ts


def process_file(filepath, plots_dir):
    with open(filepath, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "points" in data:
        points = data["points"]
        is_wrapped = True
    else:
        points = data
        is_wrapped = False

    ref_points = [p for p in points if p.get("period") == "reference" and not p.get("glint", False)]
    current_points = [p for p in points if p.get("period") == "current" and not p.get("glint", False)]
    current_points.sort(key=lambda p: parse_timestamp(p["timestamp"]))

    scored_current_points = []
    for p in current_points:
        z = point_z_score_median(p, ref_points, WINDOW_HALF_WIDTH_DEG, MIN_NEIGHBORS)
        if z is None:
            # The window has too few reference points, so leave the point
            # unscored.
            continue
        p["z_score_median"] = z
        scored_current_points.append(p)

    anomalies_median, point_to_anomaly_score = group_and_score_anomalies_median(scored_current_points, FLAG_THRESHOLD)
    for p in scored_current_points:
        score = point_to_anomaly_score.get(id(p))
        if score is not None:
            p["anomaly_score_median"] = score

    # Add the new field and keep every existing field, including the glint,
    # period, and `knn.py` fields.
    if is_wrapped:
        data["anomalies_median"] = anomalies_median
    else:
        data = {"points": points, "anomalies_median": anomalies_median}

    dirpath = os.path.dirname(os.path.abspath(filepath))
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".json.tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, filepath)

    plot_file(points, filepath, plots_dir)

    return anomalies_median


def plot_file(points, filepath, plots_dir):
    """
    Save one diagnostic PNG of magnitude against equatorial phase.

    Marker shape separates reference points from current points. Color
    separates flagged current points from unflagged ones, using
    `flagged_median`. Glint points use their own marker and sit on top,
    whatever their period. The styling matches the `knn.py` plots so you can
    compare the two plots for one file side by side.
    """
    ref_non_glint = [p for p in points if p.get("period") == "reference" and not p.get("glint", False)]
    cur_non_glint = [p for p in points if p.get("period") == "current" and not p.get("glint", False)]
    glint_pts = [p for p in points if p.get("glint", False)]

    cur_flagged = [p for p in cur_non_glint if p.get("flagged_median") is True]
    cur_unflagged = [p for p in cur_non_glint if p.get("flagged_median") is not True]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    def scatter(pts, **kwargs):
        if not pts:
            return
        ax.scatter(
            [p["equatorial_phase"] for p in pts],
            [p["magnitude"] for p in pts],
            **kwargs,
        )

    scatter(ref_non_glint, marker="o", s=18, c="#39d353", alpha=0.6, label="reference", zorder=2)
    scatter(cur_unflagged, marker="^", s=32, c="#f5e642", label="current (not flagged)", zorder=3)
    scatter(cur_flagged, marker="^", s=60, c="#ff3b3b", edgecolors="white", linewidths=0.7, label="current (flagged)", zorder=4)
    scatter(glint_pts, marker="x", s=70, c="#00e5ff", linewidths=2.0, label="glint (excluded)", zorder=5)

    ax.invert_yaxis()  # A lower magnitude is brighter, so brighter points plot higher.
    ax.set_xlabel("Equatorial phase (SEPA, deg)", color="white")
    ax.set_ylabel("Magnitude", color="white")
    ax.set_title(Path(filepath).stem + "  (rolling median)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    legend = ax.legend(loc="best", fontsize=8, facecolor="black", edgecolor="white")
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()

    rel = os.path.relpath(filepath, DATA_DIR)
    out_path = Path(plots_dir) / Path(rel).with_suffix("")
    out_path = out_path.parent / (out_path.name + "_median.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    era_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "era*")))
    for era_dir in era_dirs:
        json_files = sorted(glob.glob(os.path.join(era_dir, "*.json")))
        for filepath in json_files:
            anomalies_median = process_file(filepath, PLOTS_DIR)
            print(f"{filepath}: {len(anomalies_median)} anomaly group(s) (rolling median)")


if __name__ == "__main__":
    main()