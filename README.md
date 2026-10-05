# Polynomial Regression: Geothermal Plant Assignment

Roll number: **BT2024136**

Polynomial regression models for two datasets from the assignment:

- **var1**: Power plant steam turbine optimisation (6 inputs `x1`–`x6`, target `y` = Net Power Score)
- **var2**: Subterranean thermal reservoir mapping (3 inputs `x1`–`x3`, target `y` = Thermal Anomaly Score)

## Approach

Each model expands the inputs into polynomial terms, standardises them, and fits a regularised linear model. The degree and regularisation strength were chosen by 5-fold cross-validation on the training data.

| Problem | Model | Degree | Alpha | CV MSE | CV R² |
|---|---|---|---|---|---|
| var1 | Lasso | 5 | 0.01 | ~0.35 | ~0.963 |
| var2 | Ridge | 10 | 1.0 | ~0.24 | ~0.994 |

Notes:

- **var1:** Unregularised fits overfit quickly beyond degree 4. Lasso at degree 5 gave the lowest error and keeps only part of the 461 terms, which suggests the underlying function is sparse. Degrees 6 and 7 were slightly worse.
- **var2:** Error dropped steadily up to degree 8 and then plateaued near 0.24 for degrees 9–12. Degree 10 with ridge was chosen as the simplest model on the plateau.
- About 25–30% of input values sit exactly at -1 or 1, which looks like clipping. This probably limits how low the error can go.

CV scores are estimates from the training data. Final test scores may differ.

## Repository structure

```
.
├── train_predict.py          # trains both models, prints CV scores, writes predictions
├── README.md
├── data/                     # place the CSV files here
│   ├── BT2024136_train_var1.csv
│   ├── BT2024136_test_var1.csv
│   ├── BT2024136_train_var2.csv
│   └── BT2024136_test_var2.csv
├── BT2024136_pred_var1.csv   # output
└── BT2024136_pred_var2.csv   # output
```

## Setup

Requires Python 3.9 or newer.

```bash
pip install numpy pandas scikit-learn
```

## Usage

1. Put the four dataset CSV files in the `data/` folder.
2. Run from the repository root:

```bash
python train_predict.py
```

The script prints cross-validation MSE and R² for each problem (two different CV splits), then fits on the full training set and writes:

- `BT2024136_pred_var1.csv`
- `BT2024136_pred_var2.csv`

Each output file has a single column `y` with one prediction per test row, in the same order as the test file.

## Reproducing the model choices

To change a model, edit the `cfg` dictionary at the top of `train_predict.py`:

```python
cfg = {1: ('lasso', 5, dict(alpha=0.01, max_iter=20000, tol=1e-4)),
       2: ('ridge', 10, dict(alpha=1.0))}
```

Each entry is `(model type, polynomial degree, model parameters)`.

## Dependencies

- numpy
- pandas
- scikit-learn
