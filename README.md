# Satellite Anomaly Detection Pipeline

## Setup

```
setup.bat
venv\Scripts\activate.bat
```

- If only .csv files are present, run `csv_to_json.py` as
```
python csv_to_json.py <input_dir> data_json
```

- If using one combined .json for input run 
```
python split_eras.py <input_file> data_json
```

## Run

```
python start.py data_json output
```

- `data_json` — folder with `era1/`, `era2/`, `era3/` subfolders of per-satellite JSON files.
- `output` — folder for everything this run creates.

## Scripts run, in order

1. `glint_detect.py` — tags points near zero equatorial phase angle as glint.
2. `current_points_check.py` — labels each point as `reference` or `current` and adds file-level point counts.
3. `knn.py` — scores current points against their k-nearest reference neighbors and groups flagged points into anomalies.
4. `rolling_median.py` — scores current points against reference neighbors in a fixed SEPA window and groups flagged points into anomalies.
5. `grouping.py` — clusters satellites into normal element sets per era and checks for group-membership shifts.

## What gets created

- `data_json/era*/*.json` — updated in place with glint, period, and score fields (input files are modified, not copied).
- `output/plots_knn/era*/*.png` — one plot per satellite from the kNN scoring stage.
- `output/plots_median/era*/*.png` — one plot per satellite from the rolling-median scoring stage.
- `output/pca_plots/era*_clustering.png` and `era*_membership.json` — clustering plot and group-membership summary per era.
- `output/era1_anomalies.json`, `era2_anomalies.json`, `era3_anomalies.json` — merged kNN + rolling-median anomalies per era.
- `output/pipeline_report.json` — processed/failed counts per stage.

## For one combined final .json
```
python combine_eras.py data_json combined.json
```
