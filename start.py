"""Run the satellite anomaly-detection stages as one ordered pipeline.

The pipeline runs glint tagging, period labeling, kNN scoring,
rolling-median scoring, and normal-element-set clustering. It implements the
`run` method from Task 1 of `assignment-data-scientist-nov-2025.pdf`.

The assignment describes `run` as taking "a json file" and returning "a json
file." This pipeline takes the `data_json/` folder instead, with its
era1/era2/era3 subfolders of per-satellite files. It writes one summary JSON
per era. Two steps need the whole folder. Both scoring stages compare each
current point to the reference points of the same satellite. The Task 1d
clustering groups every satellite in an era together.

Stage order and dependencies:
    1. glint tagging: tags points. Needs no other stage.
    2. period labeling: tags points. Needs no other stage.
    3. kNN scoring: needs glint and period on every point.
    4. rolling-median scoring: needs glint and period. Does not need kNN.
    5. normal-element-set clustering: needs glint and period. Does not need
       either scoring stage.
Steps 1 and 2 can run in either order, and so can steps 3 to 5. The
pipeline runs the two tagging stages first because the other three read
their tags.

The pipeline creates `output_dir` and its `plots_knn/`, `plots_median/`,
and `pca_plots/` subfolders on the first run. Later runs reuse them. Before
the first run, only `data_dir` must exist, with at least one of era1/,
era2/, or era3/ populated.

Failure handling: the glint, period, kNN, and rolling-median stages each
process one file at a time. They catch exceptions per file, so one bad file
is logged and skipped while the rest of the era continues. Clustering must
load every satellite in an era at once to fit one set of clusters. For
that reason, an era that raises an exception is logged and skipped, and
the other eras continue. If `data_dir` is missing, the run stops at once.
"""

import argparse
import copy
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import current_points_check
import glint_detect
import grouping
import knn
import rolling_median

ERAS = ("era1", "era2", "era3")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline")


class AnomalyDetectionPipeline:
    """Run the five processing stages in order, then write one merged
    anomaly summary per era.

    `data_dir` must hold era1/, era2/, and era3/ subfolders of per-satellite
    JSON files, as `csv_to_json.py` writes them. Each stage overwrites those
    files in place. `output_dir` receives every new file from the run: the
    plots, the clustering summaries, the per-era anomaly JSON files, and a
    report of which files succeeded or failed.
    """

    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")
        self.output_dir = Path(output_dir)
        self.plots_knn_dir = self.output_dir / "plots_knn"
        self.plots_median_dir = self.output_dir / "plots_median"
        self.pca_dir = self.output_dir / "pca_plots"
        self._ensure_output_dirs()

        # The scoring stages collect anomalies here, keyed by era, so the
        # final merge does not re-read every file.
        self._knn_anomalies = {era: [] for era in ERAS}
        self._median_anomalies = {era: [] for era in ERAS}
        self.report = {"stages": {}, "started": datetime.now(timezone.utc).isoformat()}

    # -- helpers -------------------------------------------------------

    def _ensure_output_dirs(self):
        """Create each missing output folder. If a folder already exists,
        reuse it. The run overwrites only its own files and leaves other
        contents in place."""
        for d in (self.output_dir, self.plots_knn_dir, self.plots_median_dir, self.pca_dir):
            existed = d.is_dir()
            d.mkdir(parents=True, exist_ok=True)
            log.info("%s output folder: %s", "using existing" if existed else "created", d)

    def _era_dirs(self):
        """Yield (era_name, era_dir) for each era subfolder that exists."""
        for era in ERAS:
            era_dir = self.data_dir / era
            if not era_dir.is_dir():
                log.warning("skipping %s: %s not found", era, era_dir)
                continue
            yield era, era_dir

    def _run_per_file_stage(self, name, process_one):
        """Call `process_one(era, path)` for every *.json file in every
        era, in filename order. If one file raises an exception, log it
        and continue with the next file.
        """
        stats = {"processed": 0, "failed": []}
        for era, era_dir in self._era_dirs():
            for path in sorted(era_dir.glob("*.json")):
                try:
                    process_one(era, path)
                    stats["processed"] += 1
                except Exception as exc:  # Continue with the other files.
                    log.error("%s failed on %s: %s", name, path, exc)
                    stats["failed"].append({"era": era, "file": path.name, "error": str(exc)})
        self.report["stages"][name] = stats
        log.info("%s: %d processed, %d failed", name, stats["processed"], len(stats["failed"]))

    # -- stages ----------------------------------------------------------

    def _stage_glint(self):
        self._run_per_file_stage("glint_tagging", lambda era, path: glint_detect.process_file(path))

    def _stage_period(self):
        bounds_by_era = {
            era: (
                current_points_check.parse(rs),
                current_points_check.parse(re_),
                current_points_check.parse(ce),
            )
            for era, (rs, re_, ce) in current_points_check.ERAS.items()
        }

        def process_one(era, path):
            current_points_check.process_file(path, bounds_by_era[era])

        self._run_per_file_stage("period_labeling", process_one)

    def _stage_knn(self):
        # `knn.plot_file()` reads this module global to build each plot path
        # relative to the data root. Set it before the first file runs.
        knn.DATA_DIR = str(self.data_dir)

        def process_one(era, path):
            anomalies = knn.process_file(str(path), str(self.plots_knn_dir))
            self._knn_anomalies[era].extend(anomalies)

        self._run_per_file_stage("knn_scoring", process_one)

    def _stage_rolling_median(self):
        rolling_median.DATA_DIR = str(self.data_dir)  # See the knn.DATA_DIR comment above.

        def process_one(era, path):
            anomalies = rolling_median.process_file(str(path), str(self.plots_median_dir))
            self._median_anomalies[era].extend(anomalies)

        self._run_per_file_stage("rolling_median_scoring", process_one)

    def _stage_grouping(self):
        """Cluster all satellites in each era together. This stage cannot
        skip a single file, so if an era fails, log it and continue with
        the next era."""
        stats = {"processed": 0, "failed": []}
        for era, era_dir in self._era_dirs():
            try:
                grouping.process_era(era_dir, era, str(self.pca_dir))
                stats["processed"] += 1
            except Exception as exc:
                log.error("grouping failed on %s: %s", era, exc)
                stats["failed"].append({"era": era, "error": str(exc)})
        self.report["stages"]["grouping"] = stats
        log.info("grouping: %d era(s) processed, %d failed", stats["processed"], len(stats["failed"]))

    # -- final merge -------------------------------------------------------

    @staticmethod
    def merge_anomalies(knn_list, median_list):
        """Combine the kNN and rolling-median anomalies for one era.

        If two entries share a satellite and their timestamp ranges overlap,
        merge them into one entry. The code compares timestamps as strings.
        Every file uses the same ISO 8601 format
        ("2025-10-31T18:10:57.991000Z"), so string order matches time order.

        A merged entry keeps the wider `equatorial_phase` and `timestamp`
        ranges, the higher score, and the set of methods that flagged it.
        This method does not update `n_points`. A point that both methods
        flag appears in both counts, so a sum counts it twice.
        `_write_era_summaries` replaces `n_points` with an exact count.
        """
        tagged = [{**copy.deepcopy(a), "methods": ["knn"]} for a in knn_list]
        tagged += [{**copy.deepcopy(a), "methods": ["rolling_median"]} for a in median_list]

        by_norad = {}
        for a in tagged:
            by_norad.setdefault(a["norad_id"], []).append(a)

        merged = []
        for group in by_norad.values():
            group.sort(key=lambda a: a["timestamp"][0])
            current = None
            for a in group:
                if current is not None and a["timestamp"][0] <= current["timestamp"][1]:
                    current["timestamp"][0] = min(current["timestamp"][0], a["timestamp"][0])
                    current["timestamp"][1] = max(current["timestamp"][1], a["timestamp"][1])
                    current["equatorial_phase"][0] = min(current["equatorial_phase"][0], a["equatorial_phase"][0])
                    current["equatorial_phase"][1] = max(current["equatorial_phase"][1], a["equatorial_phase"][1])
                    current["score"] = max(current["score"], a["score"])
                    current["methods"] = sorted(set(current["methods"]) | set(a["methods"]))
                else:
                    if current is not None:
                        merged.append(current)
                    current = a
            if current is not None:
                merged.append(current)
        merged.sort(key=lambda a: a["score"], reverse=True)
        return merged

    def _count_flagged_points(self, era, anomaly):
        """Count the points in one merged anomaly's time range that either
        method flagged. Each point counts once.

        `merge_anomalies` cannot count them. Each input anomaly stores only
        a point count and a timestamp range, not the points themselves. This
        method re-reads the satellite's scored file instead. Each point there
        has the kNN `flagged` field and the rolling-median `flagged_median`
        field.
        """
        path = self.data_dir / era / f"{anomaly['norad_id']}.json"
        with open(path) as f:
            data = json.load(f)
        points = data["points"] if isinstance(data, dict) else data
        lo, hi = anomaly["timestamp"]
        return sum(
            1 for p in points
            if p.get("period") == "current" and not p.get("glint")
            and lo <= p["timestamp"] <= hi
            and (p.get("flagged") or p.get("flagged_median"))
        )

    def _write_era_summaries(self):
        for era in ERAS:
            merged = self.merge_anomalies(self._knn_anomalies[era], self._median_anomalies[era])
            for a in merged:
                a["n_points"] = self._count_flagged_points(era, a)
            out_path = self.output_dir / f"{era}_anomalies.json"
            out_path.write_text(json.dumps(merged, indent=2, default=str))
            log.info("%s: wrote %d merged anomalies to %s", era, len(merged), out_path)

    # -- entry point -------------------------------------------------------

    def run(self):
        """Run every stage in order and write the per-era anomaly summaries.

        Returns the report dict, which lists how many files each stage
        processed and which ones failed. The method also writes the report
        to `<output_dir>/pipeline_report.json`.
        """
        found = [era for era in ERAS if (self.data_dir / era).is_dir()]
        if not found:
            raise FileNotFoundError(
                f"No era1/era2/era3 folders found under {self.data_dir} -- nothing to process"
            )
        log.info("found era folder(s): %s", ", ".join(found))

        for stage in (
            self._stage_glint,
            self._stage_period,
            self._stage_knn,
            self._stage_rolling_median,
            self._stage_grouping,
        ):
            stage()
        self._write_era_summaries()
        self.report["finished"] = datetime.now(timezone.utc).isoformat()
        report_path = self.output_dir / "pipeline_report.json"
        report_path.write_text(json.dumps(self.report, indent=2))
        log.info("pipeline report written to %s", report_path)
        return self.report


def _check_merge_anomalies():
    """Check that `merge_anomalies` merges overlapping anomalies.

    Two anomalies for one satellite with overlapping times become one entry
    with the combined range and the higher score. An anomaly that does not
    overlap stays separate. This check skips `n_points`, because
    `merge_anomalies` does not compute it. `_check_count_flagged_points`
    checks the count.
    """
    knn_list = [
        {"norad_id": 1, "equatorial_phase": [-80.0, -70.0], "timestamp": ["2025-11-07 04:00:00", "2025-11-07 04:10:00"], "score": 0.9, "n_points": 4},
        {"norad_id": 2, "equatorial_phase": [10.0, 20.0], "timestamp": ["2025-11-07 05:00:00", "2025-11-07 05:05:00"], "score": 0.7, "n_points": 3},
    ]
    median_list = [
        {"norad_id": 1, "equatorial_phase": [-75.0, -65.0], "timestamp": ["2025-11-07 04:05:00", "2025-11-07 04:15:00"], "score": 0.95, "n_points": 5},
        {"norad_id": 2, "equatorial_phase": [30.0, 40.0], "timestamp": ["2025-11-07 06:00:00", "2025-11-07 06:05:00"], "score": 0.6, "n_points": 2},
    ]
    merged = AnomalyDetectionPipeline.merge_anomalies(knn_list, median_list)
    by_norad = {m["norad_id"]: [] for m in merged}
    for m in merged:
        by_norad[m["norad_id"]].append(m)
    assert len(by_norad[1]) == 1, "overlapping anomalies for norad_id 1 should merge into one"
    sat1 = by_norad[1][0]
    assert sat1["equatorial_phase"] == [-80.0, -65.0]
    assert sat1["timestamp"] == ["2025-11-07 04:00:00", "2025-11-07 04:15:00"]
    assert sat1["score"] == 0.95
    assert set(sat1["methods"]) == {"knn", "rolling_median"}
    assert len(by_norad[2]) == 2, "non-overlapping anomalies for norad_id 2 should stay separate"
    print("check passed")


def _check_count_flagged_points():
    """Check that a point that both methods flag counts once, not twice."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        era_dir = tmp / "era1"
        era_dir.mkdir(parents=True)
        points = [
            # Flagged by kNN only.
            {"period": "current", "glint": False, "timestamp": "2025-11-07T04:00:00Z", "flagged": True, "flagged_median": False},
            # Flagged by rolling-median only.
            {"period": "current", "glint": False, "timestamp": "2025-11-07T04:01:00Z", "flagged": False, "flagged_median": True},
            # Flagged by both. Counts once.
            {"period": "current", "glint": False, "timestamp": "2025-11-07T04:02:00Z", "flagged": True, "flagged_median": True},
            # Not flagged by either.
            {"period": "current", "glint": False, "timestamp": "2025-11-07T04:03:00Z", "flagged": False, "flagged_median": False},
            # Outside the anomaly's time range. Not counted.
            {"period": "current", "glint": False, "timestamp": "2025-11-07T05:00:00Z", "flagged": True, "flagged_median": True},
            # Glint. Not counted, even though both methods flag it.
            {"period": "current", "glint": True, "timestamp": "2025-11-07T04:00:30Z", "flagged": True, "flagged_median": True},
            # Reference period. Not counted.
            {"period": "reference", "glint": False, "timestamp": "2025-11-07T04:00:15Z", "flagged": True, "flagged_median": True},
        ]
        (era_dir / "1.json").write_text(json.dumps({"points": points}))

        pipeline = AnomalyDetectionPipeline(tmp, tmp / "output")
        anomaly = {"norad_id": 1, "timestamp": ["2025-11-07T04:00:00Z", "2025-11-07T04:03:00Z"]}
        count = pipeline._count_flagged_points("era1", anomaly)
        assert count == 3, f"expected 3 unique flagged points, got {count}"
    print("check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", nargs="?", help="Folder with era1/era2/era3 of per-satellite JSON files")
    parser.add_argument("output_dir", nargs="?", help="Folder for plots, clustering summaries, and per-era anomaly JSON")
    parser.add_argument("--check", action="store_true", help="Run the merge_anomalies self-test and exit")
    args = parser.parse_args()

    if args.check:
        _check_merge_anomalies()
        _check_count_flagged_points()
        return

    if not args.data_dir or not args.output_dir:
        parser.error("data_dir and output_dir are required unless --check is given")

    report = AnomalyDetectionPipeline(args.data_dir, args.output_dir).run()
    total_failed = sum(len(s.get("failed", [])) for s in report["stages"].values())
    if total_failed:
        log.warning("pipeline finished with %d failure(s) -- see pipeline_report.json", total_failed)
        sys.exit(1)


if __name__ == "__main__":
    main()