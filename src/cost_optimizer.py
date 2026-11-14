"""Select an ensemble on validation, then evaluate it once on the held-out test set."""

import ast
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import build_preprocessor

COST_FN = 10_000
COST_FP = 1_500
TAU_THEORY = COST_FP / (COST_FN + COST_FP)


def loss(y, probabilities, threshold):
    predicted = probabilities >= threshold
    fn = int(np.sum((y == 1) & ~predicted))
    fp = int(np.sum((y == 0) & predicted))
    return fn * COST_FN + fp * COST_FP, fn, fp


def main():
    val = pd.read_csv("data/val.csv")
    y_val = val["TARGET"].to_numpy()
    ensembles = np.load("results/ensemble_val_probs.npz")
    baseline = np.load("results/baseline_val_probs.npz")
    calibrated = np.load("results/calibrated_val_probs.npz")
    assert np.array_equal(y_val, ensembles["y_val"])
    assert np.array_equal(y_val, baseline["y_val"])
    assert np.array_equal(y_val, calibrated["y_val"])

    candidates = {
        "Dummy (Majority)": baseline["Dummy (Majority)"],
        "Ridge LogReg (L2)": baseline["Logistic Regression (Ridge L2)"],
        "Lasso LogReg (L1)": baseline["Logistic Regression (Lasso L1)"],
        "RBF SVM (Isotonic)": calibrated["RBF SVM (Isotonic Calibrated)"],
        "Random Forest": ensembles["Random Forest"],
        "HistGradientBoosting": ensembles["HistGradientBoosting"],
    }
    rows = []
    for name, prob in candidates.items():
        cost, fn, fp = loss(y_val, prob, TAU_THEORY)
        rows.append({
            "Model": name, "Val_PR_AUC": average_precision_score(y_val, prob),
            "Val_ROC_AUC": roc_auc_score(y_val, prob),
            "Val_Brier": brier_score_loss(y_val, prob),
            "Val_Loss_0.50": loss(y_val, prob, 0.5)[0],
            "Val_Loss_0.1304": cost, "Val_FN_0.1304": fn, "Val_FP_0.1304": fp,
        })
    summary = pd.DataFrame(rows).sort_values(["Val_Loss_0.1304", "Val_PR_AUC"], ascending=[True, False])
    summary.to_csv("results/metrics_summary.csv", index=False)
    winner = summary.iloc[0]["Model"]
    if winner != "HistGradientBoosting":
        raise RuntimeError(f"Winner changed to {winner}; review model reconstruction before test evaluation")

    prob_val = candidates[winner]
    thresholds = np.linspace(0, 1, 1001)
    losses = np.array([loss(y_val, prob_val, t)[0] for t in thresholds])
    best_index = int(np.argmin(losses))
    tau_selected = float(thresholds[best_index])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(thresholds, losses / 1e6, color="#175a74", linewidth=2)
    ax.axvline(tau_selected, color="#c35636", label=f"Validation minimum: {tau_selected:.3f}")
    ax.axvline(0.5, color="#777", linestyle="--", label="Default: 0.50")
    ax.set(xlabel="Probability threshold", ylabel="Assumed loss (million $)",
           title="Validation loss across decision thresholds")
    ax.legend()
    fig.tight_layout()
    fig.savefig("docs/assets/final_cost_curve.png", dpi=160)
    plt.close(fig)

    params = ast.literal_eval(pd.read_csv("results/ensemble_val_metrics.csv").set_index("Model").loc[winner, "Best_Params"])
    classifier = HistGradientBoostingClassifier(random_state=42)
    classifier.set_params(**{key.removeprefix("clf__"): value for key, value in params.items()})
    model = Pipeline([("prep", build_preprocessor()), ("clf", classifier)])
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    model.fit(train.drop(columns="TARGET"), train["TARGET"])
    y_test = test["TARGET"].to_numpy()
    prob_test = model.predict_proba(test.drop(columns="TARGET"))[:, 1]
    test_cost, fn, fp = loss(y_test, prob_test, tau_selected)
    cost_default = loss(y_test, prob_test, 0.5)[0]
    report = {
        "model": winner, "threshold_selected_on_validation": tau_selected,
        "cost_fn_assumed": COST_FN, "cost_fp_assumed": COST_FP,
        "test_rows": len(y_test), "test_pr_auc": average_precision_score(y_test, prob_test),
        "test_roc_auc": roc_auc_score(y_test, prob_test),
        "test_brier": brier_score_loss(y_test, prob_test),
        "test_fn": fn, "test_fp": fp, "test_loss_selected": test_cost,
        "test_loss_0.50": cost_default, "test_savings_vs_0.50": cost_default - test_cost,
    }
    with open("results/test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    payload = {"model": winner, "selected_threshold": tau_selected,
               "cost_fn": COST_FN, "cost_fp": COST_FP,
               "y_true": y_test.tolist(), "y_prob": np.round(prob_test, 6).tolist()}
    with open("results/test_predictions.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"))
    with open("docs/assets/test_predictions.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
