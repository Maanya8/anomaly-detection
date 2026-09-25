"""Group satellites into normal element sets per era, and check whether each
satellite's current-period behavior still belongs to its reference-period
group.

Assumes each satellite's JSON file already carries `glint` (bool) and
`period` ("reference" / "current" / null) fields, as written by the earlier
pipeline stages (glint tagging, then period/counts).

Expected layout:
    DATA_DIR/era1/<norad_id>.json
    DATA_DIR/era2/<norad_id>.json
    DATA_DIR/era3/<norad_id>.json

Each file is either a plain list of points, or the wrapped structure
{"reference_point_count": N, "current_point_count": M, "points": [...]}.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- editable constants -----------------------------------------------
DATA_DIR = "data_json"   # placeholder, edit before running
PLOTS_DIR = "pca_plots"      # placeholder, edit before running

# Minimum non-glint points needed on one SEPA side (< 0 or >= 0) to fit a
# slope for that side. Below this, the fit is too noisy to trust.
MIN_POINTS_PER_ARM = 5

# Minimum SEPA span (degrees) an arm's points must cover before its slope
# is trusted. A handful of points crammed into a fraction of a degree can
# still clear MIN_POINTS_PER_ARM, but fitting a line through them and
# extrapolating it back to SEPA = 0 amplifies ordinary point-to-point
# noise into an enormous, meaningless slope and intercept.
MIN_ARM_SEPA_SPAN = 10.0

# Candidate cluster counts tried per era; the best is picked by silhouette
# score. Widen this if an era's satellites plausibly split into more than
# 6 behavior groups.
K_RANGE = range(2, 7)

# A current-period vector farther from its nearest centroid than this
# percentile of reference-period, own-cluster distances gets flagged
# "unusual" rather than just "shifted".
UNUSUAL_DISTANCE_PERCENTILE = 95

# Prints per-satellite arm-fit diagnostics (point counts, SEPA span, whether
# the fit was rejected and fell back to reference). Turn off once you trust
# the output; it's noisy on large eras.
DEBUG = False

FEATURE_NAMES = [
    "brightness_level", "residual_variability",
    "left_slope", "right_slope", "asymmetry", "near_zero_peak",
]


def load_points(filepath):
    """Return a file's points list, regardless of which of the two known
    file shapes it uses.
    """
    with open(filepath) as f:
        data = json.load(f)
    return data["points"] if isinstance(data, dict) else data


def split_period(points):
    """Split into (reference, current) points, dropping glint-tagged points
    from both. Glint points are excluded from every downstream feature, the
    same way the point-level scoring pipeline excludes them.
    """
    ref = [p for p in points if p.get("period") == "reference" and not p.get("glint")]
    cur = [p for p in points if p.get("period") == "current" and not p.get("glint")]
    return ref, cur


def fit_arm(points, side):
    """Fit magnitude = slope * SEPA + intercept for one side of the phase
    curve. `side` is 'left' (SEPA < 0) or 'right' (SEPA >= 0). Returns
    (slope, intercept), or None if there aren't enough points on that side
    to trust the fit.
    """
    if side == "left":
        arm = [p for p in points if p["equatorial_phase"] < 0]
    else:
        arm = [p for p in points if p["equatorial_phase"] >= 0]
    if len(arm) < MIN_POINTS_PER_ARM:
        return None
    x = np.array([p["equatorial_phase"] for p in arm])
    y = np.array([p["magnitude"] for p in arm])
    if x.max() - x.min() < MIN_ARM_SEPA_SPAN:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def extract_features(points, fallback_arms=None):
    """Compute the 6-feature vector for one satellite, one period
    (reference or current), from its already period-filtered, glint-free
    points.

    `fallback_arms`, if given, is {'left': (slope, intercept), 'right':
    (slope, intercept)} from the SAME satellite's reference-period fit. A
    current-period window is often only 6 hours, so it commonly covers
    just one side of SEPA = 0; without a fallback, that satellite would
    never get a current-period vector at all. When a side falls back,
    that side's slope/intercept describes reference behavior, not this
    window's measurement, so asymmetry and near_zero_peak are only fully
    "current" when both arms were actually observed this window.

    Returns None if a side can't be fit and there's no fallback for it, or
    if there are no points at all.
    """
    if not points:
        return None
    left = fit_arm(points, "left") or (fallback_arms or {}).get("left")
    right = fit_arm(points, "right") or (fallback_arms or {}).get("right")
    if left is None or right is None:
        return None
    left_slope, left_intercept = left
    right_slope, right_intercept = right

    # Residual variability: how much magnitude scatters around the two-arm
    # fit, not around a single flat median. This isolates tumble/flicker
    # noise from the phase curve's own shape, so a steep curve doesn't get
    # mistaken for a noisy one.
    predicted = np.array([
        left_slope * p["equatorial_phase"] + left_intercept
        if p["equatorial_phase"] < 0
        else right_slope * p["equatorial_phase"] + right_intercept
        for p in points
    ])
    actual = np.array([p["magnitude"] for p in points])
    residuals = actual - predicted
    # Scaled MAD, same 1.4826 convention as the point-level scoring script,
    # so this residual spread is comparable in scale to a normal std dev.
    residual_variability = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))

    brightness_level = float(np.median(actual))
    asymmetry = right_slope - left_slope
    # Near-zero brightness, read off both arms' fits rather than from real
    # points, since the glint window usually removes the points that would
    # otherwise sit right at SEPA = 0.
    near_zero_peak = (left_intercept + right_intercept) / 2

    return np.array([
        brightness_level, residual_variability,
        left_slope, right_slope, asymmetry, near_zero_peak,
    ])


def choose_k(X_scaled):
    """Pick cluster count by silhouette score over K_RANGE, clipped to
    what the sample size can support (silhouette needs at least 2 points
    per cluster to mean anything).
    """
    max_k = min(max(K_RANGE), len(X_scaled) - 1)
    candidates = [k for k in K_RANGE if k <= max_k]
    if not candidates:
        return 2  # too few satellites for a real search; fall back
    best_k, best_score = candidates[0], -1
    for k in candidates:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def process_era(era_dir, era_name, plots_dir):
    """Extract features, cluster on the reference period, check current
    period membership, and save one scatter plot plus one summary JSON for
    this era."""
    satellites = []
    for filepath in sorted(Path(era_dir).glob("*.json")):
        points = load_points(filepath)
        ref_points, cur_points = split_period(points)
        ref_features = extract_features(ref_points)
        if ref_features is None:
            print(f"  skip {filepath.name}: not enough reference points on one arm")
            continue
        # Reference arms always exist here (ref_features succeeded), so this
        # gives current-period extraction something to fall back on for
        # whichever side the current window didn't cover.
        ref_arms = {"left": fit_arm(ref_points, "left"), "right": fit_arm(ref_points, "right")}
        cur_left_own = fit_arm(cur_points, "left")
        cur_right_own = fit_arm(cur_points, "right")
        cur_features = extract_features(cur_points, fallback_arms=ref_arms)  # may be None
        partial_current = cur_features is not None and (cur_left_own is None or cur_right_own is None)
        norad_id = points[0]["norad_id"] if points else filepath.stem
        if DEBUG:
            left_pts = [p for p in cur_points if p["equatorial_phase"] < 0]
            right_pts = [p for p in cur_points if p["equatorial_phase"] >= 0]
            left_span = (max(p["equatorial_phase"] for p in left_pts) - min(p["equatorial_phase"] for p in left_pts)) if left_pts else None
            right_span = (max(p["equatorial_phase"] for p in right_pts) - min(p["equatorial_phase"] for p in right_pts)) if right_pts else None
            print(f"    {norad_id}: cur left n={len(left_pts)} span={left_span} own_fit={cur_left_own is not None} | "
                  f"cur right n={len(right_pts)} span={right_span} own_fit={cur_right_own is not None}")
        satellites.append({
            "norad_id": norad_id, "ref_features": ref_features, "cur_features": cur_features,
            "partial_current": partial_current,
        })

    if len(satellites) < 3:
        print(f"  skip {era_name}: only {len(satellites)} satellites with usable reference data")
        return

    X_ref = np.array([s["ref_features"] for s in satellites])
    scaler = StandardScaler().fit(X_ref)
    X_ref_scaled = scaler.transform(X_ref)

    k = choose_k(X_ref_scaled)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_ref_scaled)
    ref_labels = kmeans.labels_

    # Reference-side distance to each satellite's own cluster centroid,
    # used below as the baseline for what an "unusual" current distance is.
    ref_distances = np.linalg.norm(X_ref_scaled - kmeans.cluster_centers_[ref_labels], axis=1)
    unusual_threshold = np.percentile(ref_distances, UNUSUAL_DISTANCE_PERCENTILE)

    pca = PCA(n_components=2).fit(X_ref_scaled)
    ref_pca = pca.transform(X_ref_scaled)

    results = []
    cur_pca_points = []  # for plotting: (satellite index, pca point, cur_label, shifted, unusual)
    for i, sat in enumerate(satellites):
        entry = {
            "norad_id": sat["norad_id"], "ref_cluster": int(ref_labels[i]),
            "cur_cluster": None, "shifted": False, "unusual": False,
            "cur_distance_to_centroid": None, "partial_current_data": sat["partial_current"],
        }
        if sat["cur_features"] is not None:
            x_cur_scaled = scaler.transform(sat["cur_features"].reshape(1, -1))
            cur_label = int(kmeans.predict(x_cur_scaled)[0])
            cur_distance = float(np.linalg.norm(x_cur_scaled[0] - kmeans.cluster_centers_[cur_label]))
            entry["cur_cluster"] = cur_label
            entry["shifted"] = bool(cur_label != entry["ref_cluster"])
            entry["unusual"] = bool(cur_distance > unusual_threshold)
            entry["cur_distance_to_centroid"] = cur_distance
            cur_pca_points.append((i, pca.transform(x_cur_scaled)[0], cur_label,
                                    entry["shifted"], sat["partial_current"]))
        results.append(entry)

    _save_plot(era_name, plots_dir, ref_pca, ref_labels, cur_pca_points, k)

    summary_path = Path(plots_dir) / f"{era_name}_membership.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"era": era_name, "k": k, "satellites": results}, f, indent=2)

    n_shifted = sum(r["shifted"] for r in results)
    n_unusual = sum(r["unusual"] for r in results)
    print(f"  {era_name}: {len(satellites)} satellites, k={k}, "
          f"{n_shifted} shifted membership, {n_unusual} unusual current behavior")


def _save_plot(era_name, plots_dir, ref_pca, ref_labels, cur_pca_points, k):
    """Reference points colored by cluster; current points overlaid as
    triangles; an arrow from each satellite's reference point to its
    current point, red and solid where the cluster changed, gray and
    dashed otherwise."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")

    for cluster in range(k):
        mask = ref_labels == cluster
        ax.scatter(ref_pca[mask, 0], ref_pca[mask, 1], color=cmap(cluster),
                   marker="o", s=40, label=f"cluster {cluster} (reference)", alpha=0.85)

    for i, cur_point, cur_label, shifted, partial in cur_pca_points:
        # Dashed edge = at least one arm came from the reference fallback
        # (typically a 6-hour current window that only covered one side of
        # SEPA = 0), not from this window's own current-period points.
        ax.scatter(*cur_point, color=cmap(cur_label), marker="^", s=60,
                   edgecolor="black", linewidth=0.6,
                   linestyle=("--" if partial else "-"), zorder=3)
        arrow_color = "#e2555a" if shifted else "#888888"
        arrow_style = "-" if shifted else "--"
        ax.annotate("", xy=cur_point, xytext=ref_pca[i],
                    arrowprops=dict(arrowstyle="->", color=arrow_color,
                                     linestyle=arrow_style, linewidth=1.2, alpha=0.8))

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"{era_name}: normal element sets")

    handles, labels = ax.get_legend_handles_labels()
    if any(p for *_, p in cur_pca_points):
        dashed_proxy = plt.Line2D([0], [0], marker="^", color="none", markeredgecolor="black",
                                   markerfacecolor="none", linestyle="--", markersize=8,
                                   label="dashed edge = one arm from reference fallback")
        handles.append(dashed_proxy)
    ax.legend(handles=handles, loc="best", fontsize=8)
    fig.tight_layout()

    out_path = Path(plots_dir) / f"{era_name}_clustering.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    for era_name in ("era1", "era2", "era3"):
        era_dir = Path(DATA_DIR) / era_name
        if not era_dir.is_dir():
            print(f"skip {era_name}: {era_dir} not found")
            continue
        print(f"processing {era_name}...")
        process_era(era_dir, era_name, PLOTS_DIR)


if __name__ == "__main__":
    main()