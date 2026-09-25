"""Group satellites into normal element sets per era, and check whether each
satellite's current-period behavior still belongs to its reference-period
group.

Each satellite's JSON file must already have the `glint` (bool) and `period`
("reference", "current", or null) fields. `glint_detect.py` and
`current_points_check.py` write them.

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
DATA_DIR = "data_json"   # Root folder for standalone runs. start.py ignores it.
PLOTS_DIR = "pca_plots"      # Plot folder for standalone runs. start.py ignores it.

# Minimum number of non-glint points on one SEPA side (< 0 or >= 0) to fit a
# slope for that side. With fewer points, the fit is too noisy.
MIN_POINTS_PER_ARM = 5

# Minimum SEPA span, in degrees, that an arm's points must cover before the
# code trusts its slope. A few points within a fraction of a degree can pass
# MIN_POINTS_PER_ARM. A line through them, extended back to SEPA = 0, turns
# small point-to-point noise into a very large slope and intercept.
MIN_ARM_SEPA_SPAN = 10.0

# Cluster counts to try for each era. The code picks the count with the best
# silhouette score. If an era's satellites can split into more than 6
# behavior groups, widen this range.
K_RANGE = range(2, 7)

# If a current-period vector is farther from its nearest centroid than this
# percentile of reference-period, own-cluster distances, the code flags it
# "unusual".
UNUSUAL_DISTANCE_PERCENTILE = 95

# Prints arm-fit diagnostics for each satellite: point counts, SEPA span, and
# whether the fit was rejected and fell back to the reference. The output is
# long on large eras, so keep this off unless you are checking fits.
DEBUG = False

FEATURE_NAMES = [
    "brightness_level", "residual_variability",
    "left_slope", "right_slope", "asymmetry", "near_zero_peak",
]


def load_points(filepath):
    """Return a file's points list from either of the two file shapes."""
    with open(filepath) as f:
        data = json.load(f)
    return data["points"] if isinstance(data, dict) else data


def split_period(points):
    """Split points into (reference, current) and drop glint points from both.

    No feature uses glint points, which matches the point-level scoring
    stages.
    """
    ref = [p for p in points if p.get("period") == "reference" and not p.get("glint")]
    cur = [p for p in points if p.get("period") == "current" and not p.get("glint")]
    return ref, cur


def fit_arm(points, side):
    """Fit magnitude = slope * SEPA + intercept for one side of the phase
    curve. `side` is 'left' (SEPA < 0) or 'right' (SEPA >= 0).

    Returns (slope, intercept). If the side has too few points or too small
    a SEPA span, returns None.
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
    """Compute the 6-feature vector for one satellite and one period.

    `points` must already be filtered to one period, reference or current,
    with glint points removed.

    `fallback_arms` is optional. It holds {'left': (slope, intercept),
    'right': (slope, intercept)} from the reference-period fit of the same
    satellite. The current window lasts 6 hours, so it often covers only one
    side of SEPA = 0. Without a fallback, that satellite gets no
    current-period vector. If a side uses the fallback, its slope and
    intercept describe reference behavior. In that case, `asymmetry` and
    `near_zero_peak` mix reference and current data.

    Returns None if there are no points, or if a side cannot be fit and has
    no fallback.
    """
    if not points:
        return None
    left = fit_arm(points, "left") or (fallback_arms or {}).get("left")
    right = fit_arm(points, "right") or (fallback_arms or {}).get("right")
    if left is None or right is None:
        return None
    left_slope, left_intercept = left
    right_slope, right_intercept = right

    # Measure scatter around the two-arm fit, not around one flat median.
    # This separates tumble and flicker noise from the shape of the phase
    # curve, so a steep curve does not count as a noisy one.
    predicted = np.array([
        left_slope * p["equatorial_phase"] + left_intercept
        if p["equatorial_phase"] < 0
        else right_slope * p["equatorial_phase"] + right_intercept
        for p in points
    ])
    actual = np.array([p["magnitude"] for p in points])
    residuals = actual - predicted
    # Scale the MAD by 1.4826, as the scoring stages do, so this spread is
    # on the same scale as a normal standard deviation.
    residual_variability = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))

    brightness_level = float(np.median(actual))
    asymmetry = right_slope - left_slope
    # Read the brightness at SEPA = 0 from the two arm fits, not from real
    # points. The glint window usually removes the points near SEPA = 0.
    near_zero_peak = (left_intercept + right_intercept) / 2

    return np.array([
        brightness_level, residual_variability,
        left_slope, right_slope, asymmetry, near_zero_peak,
    ])


def choose_k(X_scaled):
    """Pick the cluster count in K_RANGE with the best silhouette score.

    The largest count tried is one less than the number of satellites,
    because the silhouette score needs more samples than clusters.
    """
    max_k = min(max(K_RANGE), len(X_scaled) - 1)
    candidates = [k for k in K_RANGE if k <= max_k]
    if not candidates:
        return 2  # Too few satellites to search, so use 2 clusters.
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
        # Both reference arms exist here because `ref_features` succeeded.
        # The current-period fit uses them for any side the current window
        # does not cover.
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

    # Distance from each reference vector to its own cluster centroid. These
    # distances set the threshold for an "unusual" current distance.
    ref_distances = np.linalg.norm(X_ref_scaled - kmeans.cluster_centers_[ref_labels], axis=1)
    unusual_threshold = np.percentile(ref_distances, UNUSUAL_DISTANCE_PERCENTILE)

    pca = PCA(n_components=2).fit(X_ref_scaled)
    ref_pca = pca.transform(X_ref_scaled)

    results = []
    cur_pca_points = []  # For plotting: (satellite index, pca point, cur_label, shifted, partial)
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
    """Save the PCA scatter plot for one era.

    Circles are reference vectors, colored by cluster. Triangles are
    current vectors. An arrow links each satellite's reference vector to its
    current vector. The arrow is red and solid if the cluster changed, and
    gray and dashed otherwise.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")

    for cluster in range(k):
        mask = ref_labels == cluster
        ax.scatter(ref_pca[mask, 0], ref_pca[mask, 1], color=cmap(cluster),
                   marker="o", s=40, label=f"cluster {cluster} (reference)", alpha=0.85)

    for i, cur_point, cur_label, shifted, partial in cur_pca_points:
        # A dashed edge means at least one arm came from the reference
        # fallback. This usually happens when the 6-hour current window
        # covers only one side of SEPA = 0.
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