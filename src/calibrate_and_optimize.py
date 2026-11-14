import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score

from preprocessing import build_preprocessor

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
TAU_THEORY = COST_FP / (COST_FP + COST_FN)  # ~0.1304


def compute_total_loss(y_true: np.ndarray, y_prob: np.ndarray, tau: float) -> float:
    preds = (y_prob >= tau).astype(int)
    fn = np.sum((y_true == 1) & (preds == 0))
    fp = np.sum((y_true == 0) & (preds == 1))
    return float(fn * COST_FN + fp * COST_FP)


def find_empirical_best_tau(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    taus = np.linspace(0.01, 0.99, 981)
    losses = [compute_total_loss(y_true, y_prob, t) for t in taus]
    best_idx = int(np.argmin(losses))
    return float(taus[best_idx]), float(losses[best_idx])


def main():
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")

    X_train, y_train = train_df.drop(columns=["TARGET"]), train_df["TARGET"].values
    X_val, y_val = val_df.drop(columns=["TARGET"]), val_df["TARGET"].values

    # Load uncalibrated baseline probabilities from Step 4
    base_npz = np.load("results/baseline_val_probs.npz")
    ridge_raw_probs = base_npz["Logistic Regression (Ridge L2)"]
    svm_raw_probs = base_npz["SVM (RBF Kernel)"]

    # Fit Calibrated Ridge (Sigmoid & Isotonic)
    ridge_base = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000, random_state=42)),
    ])
    ridge_iso = CalibratedClassifierCV(ridge_base, method="isotonic", cv=5)
    ridge_iso.fit(X_train, y_train)
    ridge_iso_probs = ridge_iso.predict_proba(X_val)[:, 1]

    # Fit Calibrated RBF SVM (Isotonic vs default Platt sigmoid)
    svm_base = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", SVC(kernel="rbf", C=1.0, probability=False, random_state=42)),
    ])
    print("Fitting 5-fold Isotonic Calibration on RBF SVM (~45s)...")
    svm_iso = CalibratedClassifierCV(svm_base, method="isotonic", cv=5)
    svm_iso.fit(X_train, y_train)
    svm_iso_probs = svm_iso.predict_proba(X_val)[:, 1]

    candidates = {
        "Ridge L2 (Uncalibrated)": ridge_raw_probs,
        "Ridge L2 (Isotonic Calibrated)": ridge_iso_probs,
        "RBF SVM (Platt Sigmoid)": svm_raw_probs,
        "RBF SVM (Isotonic Calibrated)": svm_iso_probs,
    }

    # 1. Generate Reliability Diagram (Calibration Curve)
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect Calibration (y = x)")

    styles = [
        ("#1f77b4", "--", "o"),
        ("#1f77b4", "-", "s"),
        ("#d62728", "--", "^"),
        ("#d62728", "-", "D"),
    ]

    records = []
    val_cal_dict = {"y_val": y_val}

    for (name, probs), (color, ls, marker) in zip(candidates.items(), styles):
        val_cal_dict[name] = probs
        frac_pos, mean_pred = calibration_curve(y_val, probs, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y_val, probs)
        pr_auc = average_precision_score(y_val, probs)
        roc_auc = roc_auc_score(y_val, probs)

        loss_05 = compute_total_loss(y_val, probs, 0.50)
        loss_theory = compute_total_loss(y_val, probs, TAU_THEORY)
        emp_tau, loss_emp = find_empirical_best_tau(y_val, probs)

        ax.plot(
            mean_pred,
            frac_pos,
            linestyle=ls,
            marker=marker,
            color=color,
            lw=1.8,
            markersize=5,
            label=f"{name} (Brier={brier:.4f})",
        )

        records.append({
            "Model": name,
            "Val_Brier": round(brier, 4),
            "Val_PR_AUC": round(pr_auc, 4),
            "Val_ROC_AUC": round(roc_auc, 4),
            "Loss_tau_0.50 ($)": int(loss_05),
            "Loss_tau_0.1304 ($)": int(loss_theory),
            "Empirical_Best_Tau": round(emp_tau, 4),
            "Min_Empirical_Loss ($)": int(loss_emp),
        })

    ax.set_xlabel("Mean Predicted Default Probability (Quantile Bins)")
    ax.set_ylabel("Observed Fraction of Actual Defaults")
    ax.set_title("Reliability Diagram: Raw vs. Isotonic Probability Calibration")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.85)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8.5)
    plt.savefig(f"{ASSETS_DIR}/calibration_curves.png")
    plt.close(fig)
    print(f"[Saved] {ASSETS_DIR}/calibration_curves.png")

    # 2. Generate Expected Financial Loss vs. Threshold Curve
    taus = np.linspace(0.01, 0.80, 300)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    for (name, probs), (color, ls, _) in zip(candidates.items(), styles):
        losses_k = [compute_total_loss(y_val, probs, t) / 1000.0 for t in taus]
        ax.plot(taus, losses_k, color=color, linestyle=ls, lw=2, label=name)

    ax.axvline(TAU_THEORY, color="#2ca02c", linestyle=":", lw=2, label=f"Theoretical Optimal τ* = {TAU_THEORY:.3f}")
    ax.axvline(0.50, color="#7f7f7f", linestyle="-.", lw=1.6, label="Naive Default Threshold τ = 0.500")

    ax.set_xlabel("Decision Threshold (τ) — Flag Client for Intervention if P(Default) ≥ τ")
    ax.set_ylabel("Validation Expected Financial Loss ($ Thousands)")
    ax.set_title("Cost-Sensitive Decision Optimization Across Probability Thresholds")
    ax.set_xlim(0.01, 0.80)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5)
    plt.savefig(f"{ASSETS_DIR}/cost_curve.png")
    plt.close(fig)
    print(f"[Saved] {ASSETS_DIR}/cost_curve.png")

    cal_df = pd.DataFrame(records)
    cal_df.to_csv("results/calibration_metrics.csv", index=False)
    np.savez("results/calibrated_val_probs.npz", **val_cal_dict)

    print("\n--- Calibration & Threshold Optimization Summary (Validation Set) ---")
    print(cal_df.to_string(index=False))


if __name__ == "__main__":
    main()
