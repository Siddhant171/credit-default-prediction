import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates domain-specific financial risk indicators row-by-row."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        eps = 1e-6

        # 1. Credit Utilization Ratios (Latest month & 6-month average)
        limit_safe = df["LIMIT_BAL"].clip(lower=1.0)
        df["UTIL_RATIO_1"] = df["BILL_AMT1"] / limit_safe
        bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
        df["UTIL_RATIO_MEAN"] = df[bill_cols].mean(axis=1) / limit_safe

        # 2. Payment-to-Bill Ratios (PAY_AMT1 pays BILL_AMT2, etc.)
        pay_ratios = []
        for i in range(1, 6):
            bill_prev = df[f"BILL_AMT{i+1}"].clip(lower=0.0)
            ratio = np.where(
                bill_prev > 0,
                df[f"PAY_AMT{i}"] / (bill_prev + eps),
                1.0,  # No positive bill owed -> borrower in good standing
            )
            pay_ratios.append(np.clip(ratio, 0.0, 5.0))
        df["PAY_TO_BILL_MEAN"] = np.mean(pay_ratios, axis=0)

        # 3. Delinquency Severity & Frequency
        pay_status_cols = [f"PAY_{i}" for i in range(1, 7)]
        df["MAX_DELINQUENCY"] = df[pay_status_cols].max(axis=1)
        df["MONTHS_DELAYED"] = (df[pay_status_cols] > 0).sum(axis=1)

        # 4. Debt Escalation Trend (Latest bill minus oldest bill, normalized by limit)
        df["DEBT_TREND_NORM"] = (df["BILL_AMT1"] - df["BILL_AMT6"]) / limit_safe

        return df


def build_preprocessor() -> Pipeline:
    """Returns an unfitted sklearn Pipeline that engineers features and scales/encodes."""
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]

    raw_numeric_cols = (
        ["LIMIT_BAL", "AGE"]
        + [f"PAY_{i}" for i in range(1, 7)]
        + [f"BILL_AMT{i}" for i in range(1, 7)]
        + [f"PAY_AMT{i}" for i in range(1, 7)]
    )

    engineered_numeric_cols = [
        "UTIL_RATIO_1",
        "UTIL_RATIO_MEAN",
        "PAY_TO_BILL_MEAN",
        "MAX_DELINQUENCY",
        "MONTHS_DELAYED",
        "DEBT_TREND_NORM",
    ]

    all_numeric_cols = raw_numeric_cols + engineered_numeric_cols

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), all_numeric_cols),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("feature_engineering", CreditFeatureEngineer()),
            ("preprocessing", column_transformer),
        ]
    )


if __name__ == "__main__":
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")

    X_train, y_train = train_df.drop(columns=["TARGET"]), train_df["TARGET"]
    X_val, y_val = val_df.drop(columns=["TARGET"]), val_df["TARGET"]

    prep = build_preprocessor()
    X_train_proc = prep.fit_transform(X_train)
    X_val_proc = prep.transform(X_val)

    feature_names = prep.named_steps["preprocessing"].get_feature_names_out()
    print(f"Processed X_train shape: {X_train_proc.shape}")
    print(f"Processed X_val shape  : {X_val_proc.shape}")
    print(f"Total model input features: {len(feature_names)}")
    print(f"Any NaNs after preprocessing: {np.isnan(X_train_proc).sum() + np.isnan(X_val_proc).sum()}")
