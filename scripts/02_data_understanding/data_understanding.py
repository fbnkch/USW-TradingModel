import pandas as pd

from base_utils import (
    load_symbols,
    basic_checks,
    data_quality_checks,
    load_symbol_df
)
from plots_intraday import plot_intraday_close
from plots_distributions import plot_return_and_volume_distributions
from plots_intraday_patterns import plot_intraday_patterns
from plots_correlations import plot_returns_correlation
from plots_breakout_domain import plot_breakout_by_time_of_day, plot_breakout_duration

# "Main" Datei (bei ausführen werden plots erzeugt)
def descriptive_statistics(symbols):
    print("=" * 70)
    print("DESKRIPTIVE STATISTIKEN (close, volume, returns)")
    print("=" * 70)

    for symbol in symbols:
        df = load_symbol_df(symbol)

        stats = {
            "close_mean": df["close"].mean(),
            "close_median": df["close"].median(),
            "close_std": df["close"].std(),
            "close_min": df["close"].min(),
            "close_max": df["close"].max(),
            "volume_mean": df["volume"].mean(),
            "volume_median": df["volume"].median(),
            "volume_std": df["volume"].std(),
            "volume_min": df["volume"].min(),
            "volume_max": df["volume"].max(),
        }

        df["return_1m"] = df["close"].pct_change()
        stats["return_mean"] = df["return_1m"].mean()
        stats["return_std"] = df["return_1m"].std()

        stats_df = pd.DataFrame([stats], index=[symbol])
        print("\n" + "-" * 50)
        print(stats_df.round(6))


def run_all():
    # 1) Basischecks (erstes Subset)
    symbols_subset = load_symbols(n=5)
    basic_checks(symbols_subset)

    # 2) Data Quality Checks
    data_quality_checks(symbols_subset)

    # 3) Deskriptive Stats für 3 Symbole
    descriptive_statistics(["AAPL", "MSFT", "NVDA"])

    # 4) Intraday Plots
    plot_intraday_close("AAPL", days=3)
    plot_intraday_close("MSFT", days=3)

    # 5) Intraday Patterns (Volumen & Volatility)
    plot_intraday_patterns("AAPL")

    # 6) Histogramme
    plot_return_and_volume_distributions("AAPL")

    # 7) Korrelationen
    plot_returns_correlation(["AAPL", "MSFT", "NVDA", "AMZN", "META"])

    # 8) Breakout-Domain (benoetigt pre_split Daten aus Step 3)
    plot_breakout_by_time_of_day("AAPL")
    plot_breakout_duration("AAPL")


if __name__ == "__main__":
    run_all()