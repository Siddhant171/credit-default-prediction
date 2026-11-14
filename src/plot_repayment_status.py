"""Plot training-set default rate by most recent repayment status."""

import matplotlib.pyplot as plt
import pandas as pd


def main():
    train = pd.read_csv("data/train.csv")
    rates = train.groupby("PAY_1")["TARGET"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(rates.index.astype(str), rates["mean"] * 100, color="#196b7a")
    ax.set(xlabel="Most recent repayment status (PAY_0 in original UCI file)",
           ylabel="Default next month (%)", title="Default rate rises with recent payment delay")
    ax.set_ylim(0, 100)
    for bar, count in zip(bars, rates["size"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={count:,}", ha="center", fontsize=7, rotation=90)
    fig.tight_layout()
    fig.savefig("docs/assets/default_by_repayment_status.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
