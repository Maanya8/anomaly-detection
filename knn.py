"""
deviation_scoring.py

Step 1 (point-level): robust z-score for each current-window point, using its
K nearest reference-window neighbors by |SEPA| distance.

    spread = sqrt( (1.4826 * MAD(neighbor magnitudes))^2 + magnitude_unc^2 )
    z = (current magnitude - median(neighbor magnitudes)) / spread

The MAD term is the reference's normal point-to-point scatter at that SEPA;
the magnitude_unc term is this one current point's own measurement error.
Combining them in quadrature (rather than just using the MAD) means a noisy
current point (large magnitude_unc) needs a bigger raw difference before it
is flagged, while a precise point is judged against the reference scatter
alone -- z reflects "how surprising is this given everything we know," not
just "how far from the reference median."

Step 2 (anomaly-level): group consecutive flagged points (same NORAD ID,
contiguous in the time-sorted current window) into one anomaly, and turn the
group's median |z| into a 0-1 score via the normal CDF:

    score = erf(median(|z|) / sqrt(2))

Glint-tagged points (glint == true) are excluded entirely: they are never
used as reference neighbors and are never scored themselves, since glinting
is a known confound the earlier glint-tagging step already identified.

Expects the file structure produced by the earlier "period + counts" script:
    {
      "reference_point_count": N,
      "current_point_count": M,
      "points": [ {norad_id, timestamp, equatorial_phase, magnitude,
                    magnitude_unc, sensor, zeroptd, glint, period}, ... ]
    }
A plain list of points (pre-period-tagging) is also accepted for robustness.

Writes back, per point:
    "z_score"        -> float, only for scored (current, non-glint) points
    "flagged"        -> bool,  only for scored points
    "anomaly_score"  -> float, only for points belonging to a grouped anomaly
and at the file level:
    "anomalies" -> [ {norad_id, equatorial_phase, timestamp, score, n_points}, ... ]

Overwrites each JSON file in place (temp file + os.replace), same pattern as
the earlier scripts. Also saves one diagnostic PNG per file to PLOTS_DIR.
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
# era1/era2/era3 subfolder layout).
PLOTS_DIR = "plots_knn"

# Number of nearest reference neighbors (by |SEPA| distance) used to build
# the local median/MAD for each current point.
#   Smaller k  -> more locally sensitive, but a noisier MAD estimate
#                 (especially near the ends of the SEPA range, where fewer
#                 reference points are nearby).
#   Larger k   -> smoother, more stable MAD, but can blur real local
#                 structure and pull in reference points less representative
#                 of that SEPA.
# Values discussed: 5, 10, 15, 20.
K_NEIGHBORS = 15

# |z| at or above this marks a point "flagged" and eligible for grouping into
# an anomaly.
#   Lower threshold  -> more sensitive, more false positives.
#   Higher threshold -> fewer, higher-confidence flags, may miss subtle
#                       drifts.
# Values discussed: 2.5, 3.0, 3.5.
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


def knn_neighbors(target_sepa, ref_points, k):
    """K reference points with smallest |SEPA - target_sepa|."""
    ranked = sorted(ref_points, key=lambda p: abs(p["equatorial_phase"] - target_sepa))
    return ranked[:k]


def point_z_score(current_point, ref_points, k):
    """
    Robust z-score for one current point against its k nearest (by SEPA)
    reference neighbors, with the current point's own magnitude_unc folded
    in via quadrature. Returns None if there are no reference neighbors at
    all (can't score).
    """
    if not ref_points:
        return None

    neighbors = knn_neighbors(current_point["equatorial_phase"], ref_points, k)
    mags = [p["magnitude"] for p in neighbors]

    median_mag = statistics_median(mags)
    mad = robust_mad(mags, median_mag)
    ref_scatter = 1.4826 * mad
    point_unc = current_point.get("magnitude_unc", 0.0) or 0.0

    spread = math.sqrt(ref_scatter ** 2 + point_unc ** 2)

    if spread == 0:
        # Zero reference scatter AND zero reported measurement uncertainty.
        # A degenerate case, not "no deviation is possible" -- fall back to
        # a tiny epsilon so an exact match still gives z == 0, and any real
        # difference produces a (very large, correctly so) z rather than a
        # ZeroDivisionError.
        spread = 1e-6

    return (current_point["magnitude"] - median_mag) / spread


def group_and_score_anomalies(scored_current_points, threshold):
    """
    scored_current_points: current, non-glint points that have a z_score,
    already sorted by timestamp.
    Groups consecutive flagged points (no un-flagged point between them) into
    anomalies and computes each anomaly's score from the median |z| of the
    group.
    Returns (anomalies, point_id_to_anomaly_score) where anomalies is a list
    of dicts ready to attach at the file level.
    """
    anomalies = []
    point_to_score = {}

    group = []

    def close_group():
        if not group:
            return
        abs_zs = [abs(p["z_score"]) for p in group]
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
        p["flagged"] = abs(p["z_score"]) >= threshold
        if p["flagged"]:
            group.append(p)
        else:
            close_group()
            group = []
    close_group()  # flush trailing group

    return anomalies, point_to_score


def parse_timestamp(ts):
    # Points are already-parsed strings in the data; used only for plot
    # ordering, so fall back to string sort if the format is unexpected.
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return ts


def process_file(filepath, plots_dir):
    with open(filepath, "r") as f:
        data = json.load(f)

    # Accept either the current {reference_point_count, current_point_count,
    # points} structure or a plain list, for robustness.
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
        z = point_z_score(p, ref_points, K_NEIGHBORS)
        if z is None:
            # No reference neighbors available at all -- leave unscored
            # rather than fabricate a z-score.
            continue
        p["z_score"] = z
        scored_current_points.append(p)

    anomalies, point_to_anomaly_score = group_and_score_anomalies(scored_current_points, FLAG_THRESHOLD)
    for p in scored_current_points:
        score = point_to_anomaly_score.get(id(p))
        if score is not None:
            p["anomaly_score"] = score

    if is_wrapped:
        data["anomalies"] = anomalies
    else:
        data = {"points": points, "anomalies": anomalies}

    # Overwrite in place via temp file, same pattern as the earlier scripts.
    dirpath = os.path.dirname(os.path.abspath(filepath))
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".json.tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, filepath)

    plot_file(points, filepath, plots_dir)

    return anomalies


def plot_file(points, filepath, plots_dir):
    """
    One diagnostic PNG per file: magnitude vs equatorial_phase, marking:
      - reference vs current (marker shape)
      - glint points (separate marker, drawn on top, regardless of period)
      - flagged vs non-flagged current points (color)
    """
    ref_non_glint = [p for p in points if p.get("period") == "reference" and not p.get("glint", False)]
    cur_non_glint = [p for p in points if p.get("period") == "current" and not p.get("glint", False)]
    glint_pts = [p for p in points if p.get("glint", False)]

    cur_flagged = [p for p in cur_non_glint if p.get("flagged") is True]
    cur_unflagged = [p for p in cur_non_glint if p.get("flagged") is not True]

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

    # Plot reference/current first, glint last (highest zorder) so glint
    # markers always sit on top and are never hidden under other points.
    scatter(ref_non_glint, marker="o", s=18, c="#39d353", alpha=0.6, label="reference", zorder=2)
    scatter(cur_unflagged, marker="o", s=32, c="#f5e642", label="current (not flagged)", zorder=3)
    scatter(cur_flagged, marker="o", s=60, c="#ff3b3b", linewidths=0.7, label="current (flagged)", zorder=4)
    scatter(glint_pts, marker="x", s=70, c="#00e5ff", linewidths=2.0, label="glint (excluded)", zorder=5)

    ax.invert_yaxis()  # magnitude: lower is brighter, conventional to invert
    ax.set_xlabel("Equatorial phase (SEPA, deg)", color="white")
    ax.set_ylabel("Magnitude", color="white")
    ax.set_title(Path(filepath).stem, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    legend = ax.legend(loc="best", fontsize=8, facecolor="black", edgecolor="white")
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()

    rel = os.path.relpath(filepath, DATA_DIR)
    out_path = Path(plots_dir) / Path(rel).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    era_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "era*")))
    for era_dir in era_dirs:
        json_files = sorted(glob.glob(os.path.join(era_dir, "*.json")))
        for filepath in json_files:
            anomalies = process_file(filepath, PLOTS_DIR)
            print(f"{filepath}: {len(anomalies)} anomaly group(s)")


if __name__ == "__main__":
    main()