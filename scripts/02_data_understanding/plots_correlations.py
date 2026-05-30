import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from base_utils import load_symbol_df, ensure_artifacts_dir, ARTIFACTS_DIR


def plot_returns_correlation(symbols):
    returns_df = pd.DataFrame()

    for symbol in symbols:
        df = load_symbol_df(symbol)
        returns_df[symbol] = df["close"].pct_change()

    # Korrelation
    corr = returns_df.corr()

    ensure_artifacts_dir()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Korrelation der 1-Minuten-Returns (NASDAQ-100 Subset)")
    out_path = ARTIFACTS_DIR / "corr_returns_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Plot gespeichert: {out_path}")


if __name__ == "__main__":
    plot_returns_correlation(["AAPL", "MSFT", "NVDA", "AMZN", "META"])