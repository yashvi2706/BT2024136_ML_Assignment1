# Polynomial Regression: Geothermal Plant Assignment

Roll number: **BT2024136**

Polynomial regression models for the two datasets in the assignment:

- **var1**: steam turbine optimisation (6 inputs `x1`-`x6`, target `y` = Net Power Score)
- **var2**: subterranean thermal reservoir mapping (3 inputs `x1`-`x3`, target `y` = Thermal Anomaly Score)

## Approach

Pipeline for both problems: `PolynomialFeatures` -> `StandardScaler` -> regulariser.

- **Degree search:** every degree is scored with 5-fold shuffled cross-validation (seed 42), with preprocessing inside the pipeline so nothing leaks between folds. var1 searches degrees 1-10, var2 searches 1-20.
- **Selection rule:** the lowest degree whose mean CV MSE is within one standard error of the minimum (one-standard-error rule).
- **var1:** Lasso (`alpha = 0.01`), because the true function appears sparse.
- **var2:** Ridge with `alpha` chosen by `RidgeCV` (21 log-spaced values in [1e-8, 1e2]).
- **Checks:** Lasso alpha sensitivity at the chosen degree, an 80/20 hold-out check, and out-of-fold diagnostic plots.

## Results

| | var1 | var2 |
|---|---|---|
| Model | Lasso, alpha = 0.01 | Ridge, RidgeCV alpha ~ 0.32 |
| Chosen degree | 5 | 10 |
| Features (non-zero) | 461 (112) | 285 (285) |
| 5-fold CV MSE | 0.3617 +/- 0.0128 | 0.2445 +/- 0.0159 |
| 5-fold CV R2 | 0.9626 | 0.9942 |
| Hold-out MSE / R2 (20%) | 0.3577 / 0.9647 | 0.2217 / 0.9942 |

Test targets are hidden, so CV and hold-out scores are the estimates of test performance.
All per-degree numbers are in `results/degree_search_var*.csv`.

## Repository structure

```
.
├── train_predict.py            # full pipeline: search, checks, final fit, predictions
├── report.tex / report.pdf     # report
├── README.md
├── dataset/                       # place the four CSV files here
├── results/                    # degree_search_var*.csv, alpha_sensitivity_var1.csv, summary.json
├── figures/                    # degree_curves_var*.pdf, diagnostics_var*.pdf
├── BT2024136_pred_var1.csv     # predictions
└── BT2024136_pred_var2.csv
```

## Setup

Python 3.9 or newer.

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage

1. Put these files in `dataset/`:
   `BT2024136_train_var1.csv`, `BT2024136_test_var1.csv`, `BT2024136_train_var2.csv`, `BT2024136_test_var2.csv`
2. Run from the repository root:

```bash
python3 train_predict.py
```

This takes a few minutes. It prints the CV result for every degree, then writes the two prediction files, the tables in `results/` and the figures in `figures/`.

Each prediction file has one column `y`, with one row per test row, in the same order as the test file.
