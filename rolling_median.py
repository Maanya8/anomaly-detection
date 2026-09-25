"""
rolling_median_scoring.py

Same point-level / anomaly-level scoring as deviation_scoring.py (the kNN
version), but with a different neighbor-selection method: instead of the k
nearest reference points by SEPA, this uses every reference point that falls
inside a fixed SEPA window around the current point ("rolling median").

    spread = sqrt( (1.4826 * MAD(neighbor magnitudes))^2 + magnitude_unc^2 )
    z_score_median = (current magnitude - median(neighbor magnitudes)) / spread

This is ADDITIVE: it does not touch anything written by the kNN script.
It reads whatever is already in each file (including z_score / flagged /
anomaly_score / "anomalies" from the kNN run, if present) and only adds:

    per point (current, non-glint, with enough neighbors in the window):
        "z_score_median"       -> float
        "flagged_median"       -> bool
        "anomaly_score_median" -> float, only if grouped into an anomaly
    at the file level:
        "anomalies_median" -> [ {norad_id, equatorial_phase, timestamp,
                                   score, n_points}, ... ]

Every previously-written field (from glint tagging, period/counts, or the
kNN scoring script) is left exactly as it was. Glint-tagged points are
excluded entirely, same as the kNN script: never used as neighbors, never
scored themselves.

Overwrites each JSON file in place (temp file + os.replace), and saves one
diagnostic PNG per file to PLOTS_DIR (suffixed "_median" so it doesn't
overwrite the kNN script's plots).
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
# EDITABLE PARAMETERS -- change these and re-run
# ---------------------------------------------------------------------------

# Root folder containing era1/, era2/, era3/ subfolders of per-satellite JSON
# files. Placeholder -- fill in the real path before running.
DATA_DIR = "data_json"

# Where diagnostic plots get saved (one PNG per satellite file, mirroring the
# era1/era2/era3 subfolder layout). Can be the same folder the kNN script
# uses -- filenames are suffixed "_median" so nothing collides.
PLOTS_DIR = "plots_median"

# Half-width, in degrees of SEPA, of the rolling window: a current point at
# SEPA=s uses every non-glint reference point with SEPA in [s - W, s + W].
#   Narrower window -> more locally specific, but may catch too few
#                       reference points in sparse regions of SEPA.
#   Wider window     -> more stable median/MAD, but blurs local structure
#                       and can pull in less-representative reference points.
# No fixed convention here -- try a few, e.g. 2.0, 5.0, 10.0 degrees, and
# compare against the kNN version's results.
WINDOW_HALF_WIDTH_DEG = 5.0

# Minimum number of reference neighbors inside the window required to
# compute a score. Below this, the local median/MAD would be unreliable, so
# the point is left unscored (no z_score_median field) rather than guessed.
MIN_NEIGHBORS = 5

# |z_score_median| at or above this marks a point "flagged" and eligible for
# grouping into an anomaly.
# Values discussed for the kNN version: 2.5, 3.0, 3.5 -- same range applies
# here, though the rolling-median window can yield a different neighbor set
# (and so different z-scores) than kNN at the same point.
FLAG_THRESHOLD = 3.0

# ---------------------------------------------------------------------------


def robust_mad(values, center):
    """Median absolute deviation around `center` (not multiplied by 1.4826)."""
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
    """All reference points with SEPA within [target_sepa - half_width,
    target_sepa + half_width]."""
    lo, hi = target_sepa - half_width, target_sepa + half_width
    return [p for p in ref_points if lo <= p["equatorial_phase"] <= hi]


def point_z_score_median(current_point, ref_points, half_width, min_neighbors):
    """
    Rolling-median robust z-score for one current point, using every
    reference neighbor inside a fixed SEPA window, with the current point's
    own magnitude_unc folded in via quadrature. Returns None if there are
    fewer than min_neighbors reference points in the window.
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
        # Degenerate case (zero reference scatter and zero reported
        # uncertainty) -- tiny epsilon so an exact match gives z == 0
        # instead of a ZeroDivisionError.
        spread = 1e-6

    return (current_point["magnitude"] - median_mag) / spread


def group_and_score_anomalies_median(scored_current_points, threshold):
    """
    Same grouping logic as the kNN script: consecutive flagged points (no
    un-flagged point between them, in time order) become one anomaly, scored
    from the median |z_score_median| of the group via the normal CDF.
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
    close_group()  # flush trailing group

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
            # Too few reference neighbors in the window -- leave unscored
            # rather than fabricate a value.
            continue
        p["z_score_median"] = z
        scored_current_points.append(p)

    anomalies_median, point_to_anomaly_score = group_and_score_anomalies_median(scored_current_points, FLAG_THRESHOLD)
    for p in scored_current_points:
        score = point_to_anomaly_score.get(id(p))
        if score is not None:
            p["anomaly_score_median"] = score

    # Additive: attach alongside whatever else is already in the file
    # (glint, period, z_score/flagged/anomaly_score/anomalies from the kNN
    # script, etc.) without touching any of it.
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
    One diagnostic PNG per file: magnitude vs equatorial_phase, marking:
      - reference vs current (marker shape)
      - glint points (separate marker, drawn on top, regardless of period)
      - flagged vs non-flagged current points, by the ROLLING-MEDIAN result
        (flagged_median), so this plot can be compared side by side with the
        kNN script's plot for the same file.
    Same black-background styling as the kNN script's plots.
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

    ax.invert_yaxis()  # magnitude: lower is brighter, conventional to invert
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