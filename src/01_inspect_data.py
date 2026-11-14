import os
import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00350/default%20of%20credit%20card%20clients.xls"
)

def main():
    print("Fetching raw UCI Credit Card Default dataset...")
    df = pd.read_excel(DATA_URL, header=1)

    # Save a local raw copy in data/ so we don't rely on internet later
    os.makedirs("data", exist_ok=True)
    raw_path = "data/raw_credit_default.csv"
    df.to_csv(raw_path, index=False)
    print(f"Saved local raw copy to {raw_path}")

    print(f"\n1. Shape (rows, columns): {df.shape}")
    print(f"2. Total missing (NaN) cells: {df.isna().sum().sum()}")

    if "ID" in df.columns:
        print(f"3. Unique IDs: {df['ID'].nunique()} (matches row count: {df['ID'].nunique() == len(df)})")

    target_col = "default payment next month"
    counts = df[target_col].value_counts()
    pcts = (df[target_col].value_counts(normalize=True) * 100).round(2)
    print("\n4. Target Distribution (0 = Non-Default, 1 = Default):")
    for k in counts.index:
        print(f"   Class {k}: {counts[k]:,} rows ({pcts[k]}%)")

    print("\n5. Categorical Feature Value Counts:")
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        print(f"   {col}: {df[col].value_counts().sort_index().to_dict()}")

    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    neg_bills = (df[bill_cols] < 0).sum().sum()
    neg_pays = (df[pay_amt_cols] < 0).sum().sum()
    print(f"\n6. Financial Edge Cases:")
    print(f"   Negative BILL_AMT cells (overpayments/credit balances): {neg_bills}")
    print(f"   Negative PAY_AMT cells: {neg_pays}")
    print(f"   LIMIT_BAL min: {df['LIMIT_BAL'].min():,}, median: {df['LIMIT_BAL'].median():,}, max: {df['LIMIT_BAL'].max():,}")

if __name__ == "__main__":
    main()
