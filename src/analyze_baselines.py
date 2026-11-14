import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

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


def plot_class_imbalance(y_train: pd.Series) -> None:
    counts = y_train.value_counts().sort_index()
    pcts = counts / len(y_train) * 100

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = ["0: Paid on Time\n(No Default)", "1: Defaulted\n(Next Month)"]
    colors = ["#2b5c8f", "#d95f02"]

    bars = ax.bar(labels, counts.values, color=colors, width=0.52, edgecolor="black", linewidth=0.7)
    for bar, count, pct in zip(bars, counts.values, pcts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 280,
            f"{count:,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.set_ylabel("Number of Clients (Training Set, n = 21,000)")
    ax.set_title("Target Class Distribution: Asymmetric Default Imbalance")
    plt.savefig(f"{ASSETS_DIR}/class_imbalance.png")
    plt.close(fig)
    print(f"[Saved] {ASSETS_DIR}/class_imbalance.png")


def plot_correlation_heatmap(X_train_raw: pd.DataFrame) -> None:
    cols = [
        "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    ]
    corr = X_train_raw[cols].corr()

    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        ax=ax,
    )
    ax.set_title("Multicollinearity Across 6-Month Repayment Status & Billing Amounts")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.savefig(f"{ASSETS_DIR}/feature_correlation.png")
    plt.close(fig)
    print(f"[Saved] {ASSETS_DIR}/feature_correlation.png")


def plot_baseline_curves(npz_path: str) -> None:
    data = np.load(npz_path)
    y_val = data["y_val"]

    models = [
        ("Dummy (Majority)", "#7f7f7f", "--"),
        ("Logistic Regression (Ridge L2)", "#1f77b4", "-"),
        ("Logistic Regression (Lasso L1)", "#2ca02c", "-"),
        ("SVM (RBF Kernel)", "#d62728", "-"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # Left: Precision-Recall Curve
    ax_pr = axes[0]
    for key, color, ls in models:
        probs = data[key]
        prec, rec, _ = precision_recall_curve(y_val, probs)
        ap = average_precision_score(y_val, probs)
        ax_pr.plot(rec, prec, label=f"{key} (PR-AUC = {ap:.3f})", color=color, linestyle=ls, lw=2)

    ax_pr.set_xlabel("Recall (Fraction of Actual Defaults Caught)")
    ax_pr.set_ylabel("Precision (Fraction of Flagged Clients Who Default)")
    ax_pr.set_title("Validation Precision-Recall Curves")
    ax_pr.set_xlim(0, 1.0)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.grid(alpha=0.25)
    ax_pr.legend(loc="upper right", fontsize=8.5)

    # Right: ROC Curve
    ax_roc = axes[1]
    for key, color, ls in models:
        probs = data[key]
        fpr, tpr, _ = roc_curve(y_val, probs)
        auc = roc_auc_score(y_val, probs)
        ax_roc.plot(fpr, tpr, label=f"{key} (ROC-AUC = {auc:.3f})", color=color, linestyle=ls, lw=2)

    ax_roc.set_xlabel("False Positive Rate (Good Clients Flagged)")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.set_title("Validation ROC Curves")
    ax_roc.set_xlim(0, 1.0)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.grid(alpha=0.25)
    ax_roc.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/baseline_pr_roc.png")
    plt.close(fig)
    print(f"[Saved] {ASSETS_DIR}/baseline_pr_roc.png")


def plot_lasso_feature_selection(X_train_proc: np.ndarray, y_train: pd.Series, feat_names: list[str]) -> None:
    lasso = LogisticRegression(
        penalty="l1",
        C=0.01,
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )
    lasso.fit(X_train_proc, y_train)
    coefs = lasso.coef_.ravel()

    clean_names = [n.replace("num__", "").replace("cat__", "") for n in feat_names]
    coef_df = pd.DataFrame({"feature": clean_names, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=True)

    kept = coef_df[coef_df["abs_coef"] >= 1e-6].copy()
    zeroed = coef_df[coef_df["abs_coef"] < 1e-6]["feature"].tolist()

    print(f"\n--- LASSO (L1, C=0.01) FEATURE SELECTION SUMMARY ---")
    print(f"Total Engineered Features : {len(clean_names)}")
    print(f"Non-Zero Features Kept    : {len(kept)}")
    print(f"Features Zeroed Out       : {len(zeroed)} -> {zeroed}")

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    colors = ["#d95f02" if c > 0 else "#2b5c8f" for c in kept["coef"]]
    ax.barh(kept["feature"], kept["coef"], color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("Standardized Log-Odds Coefficient (Orange = Higher Risk, Blue = Lower Risk)")
    ax.set_title(f"Lasso (L1) Surviving Features ({len(kept)} of {len(clean_names)} Kept)")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/lasso_coefficients.png")
    plt.close(fig)
    print(f"[Saved] {ASSETS_DIR}/lasso_coefficients.png")


def inspect_representative_errors(X_val_raw: pd.DataFrame, y_val: pd.Series, npz_path: str, tau: float = 0.13) -> None:
    data = np.load(npz_path)
    ridge_probs = data["Logistic Regression (Ridge L2)"]
    preds = (ridge_probs >= tau).astype(int)

    df_err = X_val_raw.copy()
    df_err["actual_default"] = y_val.values
    df_err["pred_prob"] = ridge_probs
    df_err["pred_label"] = preds

    fn_cases = df_err[(df_err["actual_default"] == 1) & (df_err["pred_label"] == 0)].sort_values("pred_prob", ascending=True)
    fp_cases = df_err[(df_err["actual_default"] == 0) & (df_err["pred_label"] == 1)].sort_values("pred_prob", ascending=False)

    display_cols = ["LIMIT_BAL", "AGE", "PAY_1", "PAY_2", "BILL_AMT1", "PAY_AMT1", "pred_prob"]

    print(f"\n--- ERROR ANALYSIS AT OPTIMAL THRESHOLD (tau = {tau:.2f}) ---")
    print(f"Total Validation False Negatives (Missed Defaults): {len(fn_cases)} (Cost: ${len(fn_cases)*10_000:,})")
    print("Top 3 Confident False Negatives (Looked safe on paper, defaulted anyway):")
    print(fn_cases[display_cols].head(3).to_string())

    print(f"\nTotal Validation False Positives (False Alarms): {len(fp_cases)} (Cost: ${len(fp_cases)*1_500:,})")
    print("Top 3 Confident False Positives (Looked high-risk, paid on time):")
    print(fp_cases[display_cols].head(3).to_string())


if __name__ == "__main__":
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")

    X_train_raw, y_train = train_df.drop(columns=["TARGET"]), train_df["TARGET"]
    X_val_raw, y_val = val_df.drop(columns=["TARGET"]), val_df["TARGET"]

    prep = build_preprocessor()
    X_train_proc = prep.fit_transform(X_train_raw)
    feat_names = list(prep.named_steps["preprocessing"].get_feature_names_out())

    plot_class_imbalance(y_train)
    plot_correlation_heatmap(X_train_raw)
    plot_baseline_curves("results/baseline_val_probs.npz")
    plot_lasso_feature_selection(X_train_proc, y_train, feat_names)
    inspect_representative_errors(X_val_raw, y_val, "results/baseline_val_probs.npz", tau=0.13)
