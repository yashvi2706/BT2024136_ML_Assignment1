import argparse, json, os, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROLL = "BT2024136"
SEED = 42
N_FOLDS = 5

CONFIG = {
    1: dict(name="Net Power Score (turbine)", reg="lasso", max_deg=10, lasso_alpha=0.01),
    2: dict(name="Thermal Anomaly Score (reservoir)", reg="ridge", max_deg=20)
}

def build_model(var, degree, lasso_alpha=None):
    cfg = CONFIG[var]
    if cfg["reg"] == "lasso":
        reg = Lasso(alpha=lasso_alpha if lasso_alpha is not None else cfg["lasso_alpha"], max_iter=20000, tol=1e-4)
    else:
        reg = RidgeCV(alphas=np.logspace(-8, 2, 21))
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), reg)

def n_features(d, p):
    from math import comb
    return comb(d + p, p) - 1

def degree_search(var, X, y, cv):
    rows = []
    for degree in range(1, CONFIG[var]["max_deg"] + 1):
        model = build_model(var, degree)
        res = cross_validate(model, X, y, cv=cv, return_train_score=True,
                             scoring=("neg_mean_squared_error", "r2"))
        mse = -res["test_neg_mean_squared_error"]
        row = dict(
            degree=degree,
            n_features=n_features(X.shape[1], degree),
            cv_mse=float(mse.mean()),
            cv_mse_std=float(mse.std(ddof=1)),
            cv_r2=float(res["test_r2"].mean()),
            train_mse=float(-res["train_neg_mean_squared_error"].mean()),
            train_r2=float(res["train_r2"].mean())
        )
        rows.append(row)
        print(f"  var{var} degree {degree:2d}: CV MSE {row['cv_mse']:.4f} +/- {row['cv_mse_std']:.4f} R2 {row['cv_r2']:.4f}", flush=True)
    return pd.DataFrame(rows)

def one_se_degree(tab):
    best = tab.loc[tab["cv_mse"].idxmin()]
    se = best["cv_mse_std"] / np.sqrt(N_FOLDS)
    acceptable = tab[tab["cv_mse"] <= best["cv_mse"] + se]
    return int(best["degree"]), int(acceptable["degree"].min())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "results"), exist_ok=True)
    cv = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    summary = {}

    for var, cfg in CONFIG.items():
        print(f"\n=== var{var}: {cfg['name']} ({cfg['reg']}) ===")

        tr = pd.read_csv(f"{args.data_dir}/{ROLL}_train_var{var}.csv")
        te = pd.read_csv(f"{args.data_dir}/{ROLL}_test_var{var}.csv")

        features = [c for c in tr.columns if c != "y"]
        X, y, X_test = tr[features].values, tr["y"].values, te[features].values

        assert not np.isnan(X).any() and not np.isnan(X_test).any()

        table = degree_search(var, X, y, cv)
        min_degree, selected_degree = one_se_degree(table)

        print(f"  Minimum-CV-MSE degree = {min_degree}")
        print(f"  One-SE selected degree = {selected_degree}")

        table.to_csv(f"{args.out_dir}/results/degree_search_var{var}.csv", index=False)

        row = table[table["degree"] == selected_degree].iloc[0]

        if cfg["reg"] == "lasso":
            sensitivity = []

            for alpha in (0.001, 0.003, 0.01, 0.03, 0.1):
                res = cross_validate(
                    build_model(var, selected_degree, alpha),
                    X, y, cv=cv,
                    scoring=("neg_mean_squared_error", "r2")
                )
                sensitivity.append({
                    "alpha": alpha,
                    "cv_mse": float(-res["test_neg_mean_squared_error"].mean()),
                    "cv_r2": float(res["test_r2"].mean())
                })

            pd.DataFrame(sensitivity).to_csv(
                f"{args.out_dir}/results/alpha_sensitivity_var{var}.csv",
                index=False
            )

            print(pd.DataFrame(sensitivity).round(4).to_string(index=False))

        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=SEED)

        holdout_model = build_model(var, selected_degree)
        holdout_model.fit(Xa, ya)

        hold_predictions = holdout_model.predict(Xb)
        holdout_mse = mean_squared_error(yb, hold_predictions)
        holdout_r2 = r2_score(yb, hold_predictions)

        final_model = build_model(var, selected_degree)
        final_model.fit(X, y)

        predictions = final_model.predict(X_test)

        pd.DataFrame({"y": predictions}).to_csv(
            f"{args.out_dir}/{ROLL}_pred_var{var}.csv",
            index=False
        )

        reg = final_model[-1]

        info = {
            "degree": selected_degree,
            "cv_mse": float(row["cv_mse"]),
            "cv_r2": float(row["cv_r2"]),
            "holdout_mse": float(holdout_mse),
            "holdout_r2": float(holdout_r2)
        }

        if cfg["reg"] == "lasso":
            info["alpha"] = float(cfg["lasso_alpha"])
            info["nonzero_features"] = int((reg.coef_ != 0).sum())
        else:
            info["alpha"] = float(reg.alpha_)
            info["nonzero_features"] = int(len(reg.coef_))

        summary[f"var{var}"] = info

        print("\nSummary:")
        print(json.dumps(info, indent=2))

    summary_file = f"{args.out_dir}/results/summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print("Predictions and results saved.")

if __name__ == "__main__":
    main()
