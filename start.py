"""pipeline.py

Wraps the satellite anomaly-detection scripts (glint tagging, period
labeling, kNN scoring, rolling-median scoring, and normal-element-set
clustering) into one ordered pipeline, per the assignment's "run" method
requirement (assignment-data-scientist-nov-2025.pdf, Task 1).

The assignment describes `run` as taking "a json file" and returning "a
json file." This pipeline instead takes the already-converted data_json/
folder (with its era1/era2/era3 subfolders of per-satellite files) and
writes one summary JSON per era. Two of the required steps need that
wider scope: kNN and rolling-median scoring compare each current point
to its own satellite's reference points, and the normal-element-set
clustering in task 1d groups every satellite in an era together. Neither
fits a single-file-in, single-file-out shape.

Stage order, and why it's fixed:
    1. glint tagging          -- tags points; no dependency on period/scores
    2. period labeling        -- tags points; no dependency on glint/scores
    3. kNN scoring             -- needs glint + period on every point
    4. rolling-median scoring -- needs glint + period; independent of kNN
    5. normal-element-set clustering -- needs glint + period; independent
       of both scoring stages
Steps 1 and 2 can run in either order, and so can steps 3-5. This order
keeps the two point-tagging stages first and the three stages that read
those tags after.

Output folders (output_dir itself, plus plots_knn/, plots_median/, and
pca_plots/ beneath it) are created on first use and reused untouched on
every run after that; nothing needs to exist beforehand except data_dir
with at least one of era1/, era2/, era3/ already populated.

Failure handling: a stage that touches one file at a time (glint, period,
kNN, rolling-median) catches exceptions per file, so one bad file is
logged and skipped while the rest of the era keeps processing. Clustering
loads every satellite in an era at once -- it has to, to fit one set of
clusters -- so its failure unit is the whole era: an era that raises is
logged and skipped, and other eras continue. A missing data_dir is a
setup error, not a per-file one, and stops the run immediately.
"""

import argparse
import copy
import json
import logging
import sys
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
    """Runs the five per-satellite processing scripts in order, then writes
    one merged anomaly summary per era.

    `data_dir` must already hold era1/, era2/, era3/ subfolders of
    per-satellite JSON files (the output of csv_to_json.py). Each stage
    overwrites those files in place, matching the scripts' existing
    behavior. `output_dir` collects everything this run produces new:
    plots, the clustering summaries, the per-era anomaly JSON files, and
    a report of what succeeded or failed.
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

        # Per-file anomalies collected during the scoring stages, keyed by
        # era, so the final merge doesn't need to re-read every file.
        self._knn_anomalies = {era: [] for era in ERAS}
        self._median_anomalies = {era: [] for era in ERAS}
        self.report = {"stages": {}, "started": datetime.now(timezone.utc).isoformat()}

    # -- helpers -------------------------------------------------------

    def _ensure_output_dirs(self):
        """Create every output folder this run writes to, if it doesn't
        already exist. A folder that already exists (e.g. from a previous
        run) is reused as-is -- its old contents are left alone except for
        the specific files this run overwrites."""
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
        era, in filename order. A per-file exception is logged and
        skipped rather than stopping the stage.
        """
        stats = {"processed": 0, "failed": []}
        for era, era_dir in self._era_dirs():
            for path in sorted(era_dir.glob("*.json")):
                try:
                    process_one(era, path)
                    stats["processed"] += 1
                except Exception as exc:  # one file's error must not stop the rest
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
        # plot_file() inside knn.py reads this module-global to compute each
        # plot's path relative to the data root, so it must point at this
        # run's data_dir before any file is processed.
        knn.DATA_DIR = str(self.data_dir)

        def process_one(era, path):
            anomalies = knn.process_file(str(path), str(self.plots_knn_dir))
            self._knn_anomalies[era].extend(anomalies)

        self._run_per_file_stage("knn_scoring", process_one)

    def _stage_rolling_median(self):
        rolling_median.DATA_DIR = str(self.data_dir)  # same reason as knn.DATA_DIR above

        def process_one(era, path):
            anomalies = rolling_median.process_file(str(path), str(self.plots_median_dir))
            self._median_anomalies[era].extend(anomalies)

        self._run_per_file_stage("rolling_median_scoring", process_one)

    def _stage_grouping(self):
        """Clusters an era's satellites together, so failures can't be
        isolated per file the way the other stages' can -- a failed era
        is logged and skipped, and the rest of the run continues."""
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
        """Union the kNN and rolling-median anomalies for one era, collapsing
        entries for the same satellite whose timestamp ranges overlap into
        one entry. Timestamps compare as strings: every file uses the same
        ISO 8601 format ("2025-10-31T18:10:57.991000Z"), so string order
        matches time order.

        A collapsed entry keeps the wider equatorial_phase and timestamp
        range, the higher of the two scores (the stronger signal), the
        summed point count, and the set of methods that flagged it.
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
                    current["n_points"] += a["n_points"]
                    current["methods"] = sorted(set(current["methods"]) | set(a["methods"]))
                else:
                    if current is not None:
                        merged.append(current)
                    current = a
            if current is not None:
                merged.append(current)
        merged.sort(key=lambda a: a["score"], reverse=True)
        return merged

    def _write_era_summaries(self):
        for era in ERAS:
            merged = self.merge_anomalies(self._knn_anomalies[era], self._median_anomalies[era])
            out_path = self.output_dir / f"{era}_anomalies.json"
            out_path.write_text(json.dumps(merged, indent=2, default=str))
            log.info("%s: wrote %d merged anomalies to %s", era, len(merged), out_path)

    # -- entry point -------------------------------------------------------

    def run(self):
        """Runs every stage in order and writes the per-era anomaly summaries.

        Returns the report dict (also written to
        <output_dir>/pipeline_report.json) describing how many files each
        stage processed and which ones failed.
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
    """Same-satellite, overlapping-time anomalies from the two methods
    collapse into one entry with the union range and the higher score;
    a non-overlapping anomaly stays separate."""
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
    assert sat1["n_points"] == 9
    assert set(sat1["methods"]) == {"knn", "rolling_median"}
    assert len(by_norad[2]) == 2, "non-overlapping anomalies for norad_id 2 should stay separate"
    print("check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", nargs="?", help="Folder with era1/era2/era3 of per-satellite JSON files")
    parser.add_argument("output_dir", nargs="?", help="Folder for plots, clustering summaries, and per-era anomaly JSON")
    parser.add_argument("--check", action="store_true", help="Run the merge_anomalies self-test and exit")
    args = parser.parse_args()

    if args.check:
        _check_merge_anomalies()
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