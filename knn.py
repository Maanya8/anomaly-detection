"""Score current-window points against their k nearest reference neighbors.

Step 1 computes a robust z-score for each current-window point. The
neighbors are the K reference-window points closest in |SEPA| distance.

    spread = sqrt( (1.4826 * MAD(neighbor magnitudes))^2 + magnitude_unc^2 )
    z = (current magnitude - median(neighbor magnitudes)) / spread

The MAD term measures the normal reference scatter at that SEPA. The
`magnitude_unc` term is the measurement error of the current point. The
script adds the two in quadrature. As a result, a noisy current point needs a
larger raw difference before it is flagged. A precise point is compared
against the reference scatter alone.

Step 2 groups consecutive flagged points into one anomaly. Points in a group
share a NORAD ID and are adjacent in the time-sorted current window. The
normal CDF turns the median |z| of the group into a score from 0 to 1:

    score = erf(median(|z|) / sqrt(2))

The script ignores points where `glint` is true. They are never reference
neighbors and never get a score, because the glint-tagging step already
marked them as a known confound.

The script expects the file structure that `current_points_check.py` writes:
    {
      "reference_point_count": N,
      "current_point_count": M,
      "points": [ {norad_id, timestamp, equatorial_phase, magnitude,
                    magnitude_unc, sensor, zeroptd, glint, period}, ... ]
    }
It also accepts a plain list of points.

The script writes these fields to each point:
    "z_score"        -> float, only for scored (current, non-glint) points
    "flagged"        -> bool,  only for scored points
    "anomaly_score"  -> float, only for points in a grouped anomaly
It writes this field at the file level:
    "anomalies" -> [ {norad_id, equatorial_phase, timestamp, score, n_points}, ... ]

The script overwrites each JSON file in place through a temp file and
`os.replace`. It also saves one diagnostic PNG per file to `PLOTS_DIR`.
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
# file and repeats the era1/era2/era3 subfolder layout.
PLOTS_DIR = "plots_knn"

# Number of nearest reference neighbors, by |SEPA| distance, that set the
# local median and MAD for each current point.
#   Smaller k: more sensitive to local changes, but the MAD estimate is
#              noisier. The effect is strongest near the ends of the SEPA
#              range, where fewer reference points exist.
#   Larger k:  a more stable MAD, but it can hide real local structure and
#              include reference points from a different SEPA.
# Tested values: 5, 10, 15, 20.
K_NEIGHBORS = 15

# A point with |z| at or above this value is flagged and can join an anomaly.
#   Lower threshold:  more sensitive, more false positives.
#   Higher threshold: fewer flags with higher confidence, but it can miss
#                     small drifts.
# Tested values: 2.5, 3.0, 3.5.
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


def knn_neighbors(target_sepa, ref_points, k):
    """Return the k reference points with the smallest |SEPA - target_sepa|."""
    ranked = sorted(ref_points, key=lambda p: abs(p["equatorial_phase"] - target_sepa))
    return ranked[:k]


def point_z_score(current_point, ref_points, k):
    """
    Return the robust z-score of one current point against its k nearest
    reference neighbors by SEPA. The point's own `magnitude_unc` is added in
    quadrature. If no reference points exist, return None.
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
        # Both the reference scatter and the reported uncertainty are zero.
        # Use a small epsilon to avoid a ZeroDivisionError. An exact match
        # still gives z == 0, and any real difference gives a very large z.
        spread = 1e-6

    return (current_point["magnitude"] - median_mag) / spread


def group_and_score_anomalies(scored_current_points, threshold):
    """
    Group consecutive flagged points into anomalies and score each group.

    `scored_current_points` holds current, non-glint points that have a
    `z_score`, sorted by timestamp. A group ends at the first unflagged
    point. Each group's score comes from the median |z| of its points.

    Returns (anomalies, point_id_to_anomaly_score). `anomalies` is a list of
    dicts for the file-level "anomalies" field.
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
    close_group()  # Close the last group if the data ends on a flagged point.

    return anomalies, point_to_score


def parse_timestamp(ts):
    # This value only sets the sort order. If the format is unexpected,
    # return the string so the points sort as text.
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return ts


def process_file(filepath, plots_dir):
    with open(filepath, "r") as f:
        data = json.load(f)

    # Accept the wrapped {reference_point_count, current_point_count, points}
    # structure or a plain list of points.
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
            # No reference points exist, so leave the point unscored.
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

    # Write to a temp file first so a failure cannot corrupt the original.
    dirpath = os.path.dirname(os.path.abspath(filepath))
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".json.tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, filepath)

    plot_file(points, filepath, plots_dir)

    return anomalies


def plot_file(points, filepath, plots_dir):
    """
    Save one diagnostic PNG of magnitude against equatorial phase.

    Color separates reference points from current points, and flagged
    current points from unflagged ones. Glint points use their own marker
    and sit on top, whatever their period.
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

    # Draw glint points last, with the highest zorder, so other points never
    # hide them.
    scatter(ref_non_glint, marker="o", s=18, c="#39d353", alpha=0.6, label="reference", zorder=2)
    scatter(cur_unflagged, marker="o", s=32, c="#f5e642", label="current (not flagged)", zorder=3)
    scatter(cur_flagged, marker="o", s=60, c="#ff3b3b", linewidths=0.7, label="current (flagged)", zorder=4)
    scatter(glint_pts, marker="x", s=70, c="#00e5ff", linewidths=2.0, label="glint (excluded)", zorder=5)

    ax.invert_yaxis()  # A lower magnitude is brighter, so brighter points plot higher.
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