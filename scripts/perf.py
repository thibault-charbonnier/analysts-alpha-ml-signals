import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rolling_returns_boxplot(
    cum_df: pd.DataFrame,
    window_years: int = 2,
    periods_per_year: int = 12,   # 12 si mensuel, 252 si daily, etc.
    annualize: bool = True,
    show_fliers: bool = False,
    title: str | None = None,
):
    """
    cum_df: DataFrame index=dates, values = cumulative wealth index (ex: cumprod(1+r), démarre ~1)
    """
    cum_df = cum_df.sort_index()
    w = window_years * periods_per_year

    # rolling total return over window: W_t / W_{t-w} - 1
    rr = cum_df / cum_df.shift(w) - 1

    # optional annualization
    if annualize:
        rr = (1 + rr) ** (periods_per_year / w) - 1

    cols = rr.columns.tolist()
    data = [rr[c].dropna().to_numpy() for c in cols]

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=cols, showfliers=show_fliers)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(f"{window_years}Y rolling return" + (" (annualized)" if annualize else ""))
    plt.title(title or f"Distribution of {window_years}Y rolling returns")
    plt.tight_layout()
    plt.show()

def plot_cumulative_perf(cum_df: pd.DataFrame, logy: bool = False, figsize: tuple = (12, 6), title: str | None = None, savepath: str | None = None):
    """
    Trace la perf cumulée (chaque colonne de cum_df).
    - cum_df: index = dates, colonnes = indices de richesse cumulée (ex: cumprod(1+r))
    - logy: utiliser échelle logarithmique sur l'axe y
    - savepath: si fourni, enregistre la figure
    """
    cum_df = cum_df.sort_index()
    # s'assurer que l'index est datetime
    if not np.issubdtype(cum_df.index.dtype, np.datetime64):
        try:
            cum_df.index = pd.to_datetime(cum_df.index)
        except Exception:
            pass

    plt.figure(figsize=figsize)
    for col in cum_df.columns:
        plt.plot(cum_df.index, cum_df[col], label=col)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Performance")
    if logy:
        plt.yscale("log")
    plt.title(title or "Cumulative performance")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()

def compute_backtest_metrics_from_cumulative(
        cum_df: pd.DataFrame,
        periods_per_year: int = 12,
        risk_free_rate: float = 0.0,
        as_percent: bool = True,
        round_digits: int | None = 2,
) -> pd.DataFrame:
    """
    Entrée:
      - cum_df: DataFrame index=datetime, colonnes=modèles, valeurs = cumulative wealth index (ex: cumprod(1+r))
      - periods_per_year: 12 pour monthly, 252 pour daily, ...
      - risk_free_rate: taux sans risque annualisé (décimal), utilisé pour risk-adjusted return si voulu
      - as_percent: si True renvoie les métriques en pourcentages
      - round_digits: arrondi des valeurs (None pour pas d'arrondi)

    Retour:
      DataFrame index = modèles, colonnes = ['gross_return', 'annual_return', 'annual_vol', 'risk_adjusted_return', 'max_drawdown']
    """
    cum = cum_df.copy()
    # s'assurer index datetime et trié
    try:
        cum.index = pd.to_datetime(cum.index)
    except Exception:
        pass
    cum = cum.sort_index()

    results = {}
    for col in cum.columns:
        s = cum[col].dropna()
        n = len(s)
        if n == 0:
            results[col] = [np.nan] * 5
            continue

        # gross return (final wealth - 1)
        gross_return = s.iloc[-1] - 1.0

        # annualized return (exponent = periods_per_year / n_periods)
        annual_return = s.iloc[-1] ** (periods_per_year / n) - 1.0 if s.iloc[-1] > 0 else np.nan

        # periodic returns from cumulative
        per_ret = s.pct_change().dropna()
        annual_vol = per_ret.std(ddof=0) * np.sqrt(periods_per_year) if len(per_ret) > 0 else np.nan

        # risk-adjusted return (Sharpe-like). rf annualized; convert rf to periodic if wanted, here use ann values
        risk_adj = ((annual_return - risk_free_rate) / annual_vol if (
                    not np.isnan(annual_vol) and annual_vol != 0) else np.nan )/ 100

        # max drawdown
        rolling_max = s.cummax()
        drawdown = s / rolling_max - 1.0
        max_dd = drawdown.min()  # négatif

        results[col] = [gross_return, annual_return, annual_vol, risk_adj, max_dd]

    metrics = pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=["gross_return", "annual_return", "annual_vol", "risk_adjusted_return", "max_drawdown"]
    )

    # formattage
    if as_percent:
        metrics = metrics * 100.0
    if round_digits is not None:
        metrics = metrics.round(round_digits)

    metrics.T.to_excel("outputs/backtest/metrics.xlsx", sheet_name="metrics")
