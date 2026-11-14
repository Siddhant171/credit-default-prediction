import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ASSETS_DIR = "docs/assets"
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

COST_FN = 10_000.0
COST_FP = 1_500.0
TAU_STAR = COST_FP / (COST_FP + COST_FN)  # ~0.1304


def evaluate_probs(name: str, y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds_05 = (probs >= 0.50).astype(int)
    preds_opt = (probs >= TAU_STAR).astype(int)

    fn_05 = int(np.sum((y_true == 1) & (preds_05 == 0)))
    fp_05 = int(np.sum((y_true == 0) & (preds_05 == 1)))
    loss_05 = fn_05 * COST_FN + fp_05 * COST_FP

    fn_opt = int(np.sum((y_true == 1) & (preds_opt == 0)))
    fp_opt = int(np.sum((y_true == 0) & (preds_opt == 1)))
    loss_opt = fn_opt * COST_FN + fp_opt * COST_FP

    return {
        "Model": name,
        "PR_AUC": round(float(average_precision_score(y_true, probs)), 4),
        "ROC_AUC": round(float(roc_auc_score(y_true, probs)), 4),
        "Brier": round(float(brier_score_loss(y_true, probs)), 4),
        "Recall_at_0.13": round(float(recall_score(y_true, preds_opt, zero_division=0)), 4),
        "Precision_at_0.13": round(float(precision_score(y_true, preds_opt, zero_division=0)), 4),
        "F1_at_0.13": round(float(f1_score(y_true, preds_opt, zero_division=0)), 4),
        "FN_at_0.13": fn_opt,
        "FP_at_0.13": fp_opt,
        "Loss_at_0.50 ($)": int(loss_05),
        "Loss_at_0.13 ($)": int(loss_opt),
        "Savings_vs_0.50 ($)": int(loss_05 - loss_opt),
    }


def plot_loss_comparison(df: pd.DataFrame) -> None:
    plot_df = df.sort_values("Loss_at_0.13 ($)", ascending=True).copy()
    models = plot_df["Model"].tolist()
    loss_05_k = plot_df["Loss_at_0.50 ($)"].values / 1000.0
    loss_opt_k = plot_df["Loss_at_0.13 ($)"].values / 1000.0

    y_pos = np.arange(len(models))
    height = 0.36

    fig, ax = plt.subplots(figsize=(9.5, max(4.2, 0.85 * len(models))))
    bars_05 = ax.barh(y_pos + height / 2, loss_05_k, height, label="Naive Threshold (τ = 0.50)", color="#8c969e", edgecolor="black", linewidth=0.6)
    bars_opt = ax.barh(y_pos - height / 2, loss_opt_k, height, label=f"Cost-Optimal Threshold (τ* = {TAU_STAR:.3f})", color="#1b7837", edgecolor="black", linewidth=0.6)

    for bar in bars_05:
        w = bar.get_width()
        ax.text(w + 80, bar.get_y() + bar.get_height() / 2, f"${w:,.0f}k", va="center", fontsize=8.5, color="#444444")

    for bar in bars_opt:
        w = bar.get_width()
        ax.text(w + 80, bar.get_y() + bar.get_height() / 2, f"${w:,.0f}k", va="center", fontsize=8.5, fontweight="bold", color="#1b7837")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlim(0, max(loss_05_k) * 1.16)
    ax.set_xlabel("Validation Expected Financial Loss ($ Thousands — Lower is Better)")
    ax.set_title("Model Selection Leaderboard: Naive (τ = 0.50) vs. Cost-Optimal (τ* = 0.1304)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/model_loss_comparison.png")
    plt.close(fig)
    print(f"[Saved] {ASSETS_DIR}/model_loss_comparison.png")


def main():
    base_npz = np.load("results/baseline_val_probs.npz")
    cal_npz = np.load("results/calibrated_val_probs.npz")
    y_val = base_npz["y_val"]

    all_probs = {
        "Dummy (Majority)": base_npz["Dummy (Majority)"],
        "Lasso LogReg (L1)": base_npz["Logistic Regression (Lasso L1)"],
        "Ridge LogReg (L2)": cal_npz["Ridge L2 (Uncalibrated)"],
        "RBF SVM (Isotonic)": cal_npz["RBF SVM (Isotonic Calibrated)"],
    }

    # Automatically load Siddhantkumar's tree ensembles once pushed
    ensemble_path = "results/ensemble_val_probs.npz"
    if os.path.exists(ensemble_path):
        ens_npz = np.load(ensemble_path)
        for key in ens_npz.files:
            if key != "y_val":
                all_probs[key] = ens_npz[key]
        print(f"[Loaded] Found ensemble predictions in {ensemble_path}: {[k for k in ens_npz.files if k != 'y_val']}")
    else:
        print(f"[Waiting] {ensemble_path} not found yet — generating leaderboard for current models.")

    rows = [evaluate_probs(name, y_val, probs) for name, probs in all_probs.items()]
    leaderboard = pd.DataFrame(rows).sort_values(by=["Loss_at_0.13 ($)", "PR_AUC"], ascending=[True, False])

    leaderboard.to_csv("results/val_leaderboard.csv", index=False)
    plot_loss_comparison(leaderboard)

    print("\n=== UNIFIED VALIDATION LEADERBOARD (RANKED BY EXPECTED DOLLAR LOSS) ===")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
