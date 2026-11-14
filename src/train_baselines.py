import os
import sys
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import build_preprocessor

COST_FN = 10000.0  # Approving a borrower who defaults
COST_FP = 1500.0   # Rejecting a borrower who would pay on time


def expected_loss_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(fn * COST_FN + fp * COST_FP)


def main():
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")

    X_train, y_train = train_df.drop(columns=["TARGET"]), train_df["TARGET"].values
    X_val, y_val = val_df.drop(columns=["TARGET"]), val_df["TARGET"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tau_star = COST_FP / (COST_FP + COST_FN)  # Theoretical optimal threshold (~0.1304)

    models = {}

    # 1. Majority-Class Dummy Baseline
    dummy_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", DummyClassifier(strategy="most_frequent")),
    ])
    dummy_pipe.fit(X_train, y_train)
    models["Dummy (Majority)"] = dummy_pipe

    # 2. Ridge Logistic Regression (L2) - tuned via 5-fold CV on PR-AUC
    ridge_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)),
    ])
    ridge_grid = GridSearchCV(
        ridge_pipe,
        param_grid={"clf__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
    )
    ridge_grid.fit(X_train, y_train)
    print(f"Best Ridge C: {ridge_grid.best_params_['clf__C']} (CV PR-AUC: {ridge_grid.best_score_:.4f})")
    models["Logistic Regression (Ridge L2)"] = ridge_grid.best_estimator_

    # 3. Lasso Logistic Regression (L1) - tuned via 5-fold CV on PR-AUC
    lasso_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, random_state=42)),
    ])
    lasso_grid = GridSearchCV(
        lasso_pipe,
        param_grid={"clf__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
    )
    lasso_grid.fit(X_train, y_train)
    best_lasso = lasso_grid.best_estimator_
    coefs = best_lasso.named_steps["clf"].coef_[0]
    zero_count = np.sum(np.abs(coefs) < 1e-6)
    print(
        f"Best Lasso C: {lasso_grid.best_params_['clf__C']} "
        f"(CV PR-AUC: {lasso_grid.best_score_:.4f}, zeroed features: {zero_count}/{len(coefs)})"
    )
    models["Logistic Regression (Lasso L1)"] = best_lasso

    # 4. Support Vector Machine (RBF Kernel)
    svm_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
    ])
    print("Training RBF SVM (may take ~30-60 seconds for Platt probability calibration)...")
    svm_pipe.fit(X_train, y_train)
    models["SVM (RBF Kernel)"] = svm_pipe

    # Evaluate all models on Validation Set
    os.makedirs("results", exist_ok=True)
    records = []
    val_probs = {"y_val": y_val}

    for name, est in models.items():
        probs = est.predict_proba(X_val)[:, 1]
        val_probs[name] = probs

        pr_auc = average_precision_score(y_val, probs)
        roc_auc = roc_auc_score(y_val, probs)
        brier = brier_score_loss(y_val, probs)
        loss_05 = expected_loss_at_threshold(y_val, probs, 0.50)
        loss_opt = expected_loss_at_threshold(y_val, probs, tau_star)

        records.append({
            "Model": name,
            "Val_PR_AUC": round(pr_auc, 4),
            "Val_ROC_AUC": round(roc_auc, 4),
            "Val_Brier": round(brier, 4),
            "Loss_at_0.50 ($)": int(loss_05),
            "Loss_at_0.13 ($)": int(loss_opt),
        })

    results_df = pd.DataFrame(records)
    results_df.to_csv("results/baseline_val_metrics.csv", index=False)
    np.savez("results/baseline_val_probs.npz", **val_probs)

    print("\n--- Validation Results (Baselines & Linear/Margin Models) ---")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
