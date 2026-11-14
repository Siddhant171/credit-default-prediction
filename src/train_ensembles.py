"""Tune tree ensembles on training data and export validation probabilities."""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import build_preprocessor


def main() -> None:
    train = pd.read_csv("data/train.csv")
    val = pd.read_csv("data/val.csv")
    x_train, y_train = train.drop(columns="TARGET"), train["TARGET"].to_numpy()
    x_val, y_val = val.drop(columns="TARGET"), val["TARGET"].to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
            {"clf__max_depth": [8, None], "clf__min_samples_leaf": [2, 10]},
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(random_state=42),
            {"clf__max_leaf_nodes": [15, 31], "clf__learning_rate": [0.05, 0.1]},
        ),
    }

    probabilities = {"y_val": y_val}
    metrics = []
    importances = []
    for name, (classifier, grid) in candidates.items():
        search = GridSearchCV(
            Pipeline([("prep", build_preprocessor()), ("clf", classifier)]),
            param_grid=grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        print(f"Tuning {name} with five training folds...", flush=True)
        search.fit(x_train, y_train)
        model = search.best_estimator_
        prob = model.predict_proba(x_val)[:, 1]
        probabilities[name] = prob
        metrics.append({
            "Model": name,
            "Best_Params": str(search.best_params_),
            "CV_PR_AUC": search.best_score_,
            "Val_PR_AUC": average_precision_score(y_val, prob),
            "Val_ROC_AUC": roc_auc_score(y_val, prob),
            "Val_Brier": brier_score_loss(y_val, prob),
        })

        if name == "Random Forest":
            names = model.named_steps["prep"].named_steps["preprocessing"].get_feature_names_out()
            importances.extend(
                {"Model": name, "Feature": feature, "Importance": float(value)}
                for feature, value in zip(names, model.named_steps["clf"].feature_importances_)
            )
        print(f"{name}: CV PR-AUC {search.best_score_:.4f}; validation PR-AUC {metrics[-1]['Val_PR_AUC']:.4f}", flush=True)

    os.makedirs("results", exist_ok=True)
    np.savez_compressed("results/ensemble_val_probs.npz", **probabilities)
    pd.DataFrame(metrics).to_csv("results/ensemble_val_metrics.csv", index=False)
    pd.DataFrame(importances).sort_values("Importance", ascending=False).to_csv(
        "results/random_forest_importances.csv", index=False
    )
    print("Saved validation probabilities, metrics, and Random Forest importances.")


if __name__ == "__main__":
    main()
