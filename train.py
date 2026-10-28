"""Train + compare models, log to MLflow, save the best classifier & regressor.
Usage: python train.py --data india_housing_prices.csv [--sample 100000]
"""
import argparse, json, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd, mlflow, mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from features import CAT, NUM, FEATURES, HORIZON, clean, add_derived, good_investment_label, growth_rate, future_price

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
SEED = 42


def pre(scale):
    return ColumnTransformer([("num", StandardScaler() if scale else "passthrough", NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20), CAT)])


def top_features(pipe, k=15):
    names = [n.split("__", 1)[1] for n in pipe.named_steps["pre"].get_feature_names_out()]
    m = pipe.named_steps["model"]
    imp = m.feature_importances_ if hasattr(m, "feature_importances_") else np.abs(m.coef_).ravel()
    s = pd.Series(imp, index=names).sort_values(ascending=False).head(k)
    return {k_: float(v) for k_, v in s.items()}


def main(a):
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("real_estate_advisor")

    # ---------- 1. preprocessing ----------
    df = clean(pd.read_csv(a.data))
    if a.sample and len(df) > a.sample:
        df = df.sample(a.sample, random_state=SEED)
    city_med = (df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]).groupby(df["City"]).median().to_dict()
    overall_med = float((df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]).median())
    df = add_derived(df, city_med, overall_med)
    # outliers: trim extreme price/sqft
    lo, hi = df["Price_per_SqFt"].quantile([0.005, 0.995]); df = df[df["Price_per_SqFt"].between(lo, hi)].copy()
    infra_median = float(df["Infra_Score"].median())
    df["Good_Investment"] = good_investment_label(df, infra_median)
    df["Annual_Growth"] = growth_rate(df, np.random.default_rng(SEED))
    df["Future_Price_5Y"] = future_price(df["Price_in_Lakhs"], df["Annual_Growth"])
    df.to_csv(ROOT / "cleaned_dataset.csv", index=False)
    print(f"rows={len(df)}  good-investment rate={df['Good_Investment'].mean():.2%}")

    X = df[FEATURES]; yc = df["Good_Investment"]; yr = df["Annual_Growth"]
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=yc)
    va, te = train_test_split(tmp, test_size=0.5, random_state=SEED, stratify=yc.iloc[tmp])
    Xtr, Xva, Xte = X.iloc[tr], X.iloc[va], X.iloc[te]

    # ---------- 2. classification ----------
    clf_models = {
        "LogisticRegression": Pipeline([("pre", pre(True)), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "RandomForest": Pipeline([("pre", pre(False)), ("model", RandomForestClassifier(n_estimators=150, max_depth=16, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=SEED))]),
        "XGBoost": Pipeline([("pre", pre(False)), ("model", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=SEED))]),
    }
    def cls_metrics(p, Xs, ys):
        pr, pp = p.predict(Xs), p.predict_proba(Xs)[:, 1]
        return {"accuracy": accuracy_score(ys, pr), "f1": f1_score(ys, pr), "precision": precision_score(ys, pr),
                "recall": recall_score(ys, pr), "roc_auc": roc_auc_score(ys, pp)}
    clf_res, runs = {}, {}
    for name, p in clf_models.items():
        with mlflow.start_run(run_name=f"clf_{name}") as run:
            p.fit(Xtr, yc.iloc[tr])
            v, t = cls_metrics(p, Xva, yc.iloc[va]), cls_metrics(p, Xte, yc.iloc[te])
            mlflow.log_param("task", "classification"); mlflow.log_param("model", name)
            mlflow.log_metrics({**{f"val_{k}": x for k, x in v.items()}, **{f"test_{k}": x for k, x in t.items()}})
            cm = confusion_matrix(yc.iloc[te], p.predict(Xte)).tolist()
            mlflow.log_dict({"confusion_matrix": cm}, "confusion_matrix.json")
            clf_res[name] = {"val": v, "test": t, "confusion_matrix": cm}; runs[("clf", name)] = run.info.run_id
        print(f"[clf] {name:20s} val F1={v['f1']:.4f}  test F1={t['f1']:.4f}  acc={t['accuracy']:.4f}")
    best_c = max(clf_res, key=lambda k: clf_res[k]["val"]["f1"])

    # ---------- 3. regression (annual growth -> 5y price) ----------
    reg_models = {
        "Ridge": Pipeline([("pre", pre(True)), ("model", Ridge(alpha=1.0))]),
        "RandomForest": Pipeline([("pre", pre(False)), ("model", RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_leaf=3, n_jobs=-1, random_state=SEED))]),
        "XGBoost": Pipeline([("pre", pre(False)), ("model", XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.08, n_jobs=-1, random_state=SEED))]),
    }
    def reg_metrics(p, Xs, idxs):
        rate = p.predict(Xs); price = Xs["Price_in_Lakhs"].values
        pf, tf = future_price(price, rate), df["Future_Price_5Y"].iloc[idxs].values
        return {"rmse_price": float(np.sqrt(mean_squared_error(tf, pf))), "mae_price": mean_absolute_error(tf, pf),
                "r2_price": r2_score(tf, pf), "rmse_rate": float(np.sqrt(mean_squared_error(yr.iloc[idxs], rate)))}
    reg_res = {}
    for name, p in reg_models.items():
        with mlflow.start_run(run_name=f"reg_{name}") as run:
            p.fit(Xtr, yr.iloc[tr])
            v, t = reg_metrics(p, Xva, va), reg_metrics(p, Xte, te)
            mlflow.log_param("task", "regression"); mlflow.log_param("model", name)
            mlflow.log_metrics({**{f"val_{k}": x for k, x in v.items()}, **{f"test_{k}": x for k, x in t.items()}})
            reg_res[name] = {"val": v, "test": t}; runs[("reg", name)] = run.info.run_id
        print(f"[reg] {name:20s} val RMSE={v['rmse_price']:.2f}L  test RMSE={t['rmse_price']:.2f}L  R2={t['r2_price']:.4f}")
    best_r = min(reg_res, key=lambda k: reg_res[k]["val"]["rmse_price"])

    # ---------- 4. save + register best ----------
    for kind, name, pipes, fn in [("clf", best_c, clf_models, "classifier"), ("reg", best_r, reg_models, "regressor")]:
        joblib.dump(pipes[name], MODELS / f"{fn}.joblib")
        try:
            with mlflow.start_run(run_id=runs[(kind, name)]):
                mlflow.sklearn.log_model(pipes[name], name="model", serialization_format="cloudpickle", registered_model_name=f"RealEstate_{fn}")
        except Exception as e:
            print("MLflow model registry skipped:", e)
    opts = {c: sorted(df[c].unique().tolist()) for c in CAT if c not in ("State", "City")}
    meta = {"city_median_pps": city_med, "overall_median_pps": overall_med, "infra_median": infra_median,
            "best_classifier": best_c, "best_regressor": best_r, "horizon": HORIZON,
            "classification": clf_res, "regression": reg_res,
            "importance_clf": top_features(clf_models[best_c]), "importance_reg": top_features(reg_models[best_r]),
            "city_by_state": {s: sorted(g["City"].unique().tolist()) for s, g in df.groupby("State")},
            "options": opts, "ranges": {"size": [int(df.Size_in_SqFt.min()), int(df.Size_in_SqFt.max())],
                                         "price": [float(df.Price_in_Lakhs.min()), float(df.Price_in_Lakhs.max())]}}
    json.dump(meta, open(MODELS / "meta.json", "w"), indent=2)
    keep = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
            "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Public_Transport_Accessibility", "Furnished_Status",
            "Facing", "Parking_Space", "Amenity_Count", "Availability_Status", "Good_Investment", "Future_Price_5Y"]
    df[keep].sample(min(40000, len(df)), random_state=SEED).to_csv(MODELS / "sample.csv.gz", index=False)
    print(f"\nBest classifier: {best_c} | Best regressor: {best_r} | saved to {MODELS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="india_housing_prices.csv")
    ap.add_argument("--sample", type=int, default=0, help="subsample rows for faster training"); main(ap.parse_args())
