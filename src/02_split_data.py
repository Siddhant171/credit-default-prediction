import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    raw_path = "data/raw_credit_default.csv"
    df = pd.read_csv(raw_path)

    # 1. Drop ID (arbitrary identifier) and standardize target/column names
    df = df.drop(columns=["ID"])
    df = df.rename(columns={
        "default payment next month": "TARGET",
        "PAY_0": "PAY_1"
    })

    # 2. Clean undocumented categorical codes (row-level, zero data leakage)
    # EDUCATION: 1=grad, 2=univ, 3=high school, 4=others (recode 0, 5, 6 -> 4)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x in [1, 2, 3] else 4)
    # MARRIAGE: 1=married, 2=single, 3=others (recode 0 -> 3)
    df["MARRIAGE"] = df["MARRIAGE"].apply(lambda x: x if x in [1, 2] else 3)

    # 3. Stratified 70% Train / 15% Validation / 15% Test split (random_state=42)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["TARGET"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["TARGET"], random_state=42
    )

    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("Cleaned categorical codes:")
    print("  EDUCATION:", df["EDUCATION"].value_counts().sort_index().to_dict())
    print("  MARRIAGE :", df["MARRIAGE"].value_counts().sort_index().to_dict())

    print("\nSaved fixed splits to data/:")
    for name, split in [("train.csv", train_df), ("val.csv", val_df), ("test.csv", test_df)]:
        print(f"  {name:<10} -> {split.shape[0]:,} rows, {split.shape[1]} cols | Default rate: {split['TARGET'].mean()*100:.2f}%")

if __name__ == "__main__":
    main()
