import polars as pl
from datetime import date, datetime
from .utils import _validate_date

DateLike = date | str | datetime | pl.Date

class BookEngine:
    """
    High-level module responsible of building analysts's book (long-short portfolio)
    implied from the forcasted returns of the covered stocks.

    The engine is based and extensively use polars to allow high-performance
    computations with very large datasets.

    Book building methodology at time t is detailed bellow:
        - Perform a temporal filter on target-prices (t - validity_length < t_tp < t)
        - Join target-price information with market prices
        - Computed implied forcasted returns (R_i_t = TP_i_t / P_i_t - 1)
          Please note that the implied return is built with current prices at t
        - Perform a decay on timplied forcasted returns to lower the conviction
          on old target prices. The implied weights will be smaller in absolute value
        - Compute the implied weights for each stock in analyst coverage (long-short)
        - Eventualy join with other information such as stock metadata
    """
    def __init__(
        self,
        *,
        df_tp: pl.DataFrame,
        df_prices: pl.DataFrame,
        df_metadata: pl.DataFrame = None,
        validity_length: int = 12,
        decay_half_life: int = 6
    ) -> None:
        """
        Parameters
        ----------
        df_tp: polars.DataFrame
            DataFrame with analysts's target prices informations

        df_prices: polars.DataFrame
            DataFrame with market prices informations

        df_metadata: polars.DataFrame, default None
            DataFrame with stock metadata informations

        validity_length: int, default 12
            Number of month for which a target price is considered still valid

        decay_half_life: int, default 6
            Parameter of the exponential decay, number of month needed to
            decrease the power of the recommandation by 2
        """
        self.df_tp = df_tp
        self.df_prices = df_prices
        self.df_metadata = df_metadata
        self.validity_length = validity_length
        self.decay_half_life = decay_half_life

    def at_snapshot(self, snapshot_date: DateLike) -> pl.DataFrame:
        """
        Build analysts's books at a specific snapshot_date.

        Parameters
        ----------
        snapshort_date: date, datetime, pl.Date or string
            Date for which the engine will compute analysts's implied books

        Returns
        -------
        polars.DataFrame
            DataFrame containing the implied portfolios
        """
        t = _validate_date(snapshot_date)
        df_tp_valid = self._filter_valid_tp(self.df_tp, t)

        # A adapter quand on aura le dataset des prix
        df_full = df_tp_valid.join(self.df_prices, on="symbol", how="left")
        
        df_imp_returns = self._compute_imp_returns(df_full)
        df_decayed = self._apply_decay(df_imp_returns)

        df_weights = self._compute_weights(df_decayed)
        return (
            df_weights.join(self.df_metadata, on="symbol", how="left")
            if self.df_metadata is not None else df_weights
        )

    def _filter_valid_tp(self, df_tp: pl.DataFrame, t: pl.Date) -> pl.DataFrame:
        """
        Perform a temporal filter on target-prices (t - validity_length < t_tp < t).

        Parameters
        ----------
        df_tp: polars.DataFrame
            DataFrame with analysts's target prices informations
        t: polars.Date
            Snapshot date

        Returns
        -------
        polars.DataFrame
            DataFrame containing only valid target prices
        """
        max_days_valid = self.validity_length * 30.4
        return (
            df_tp
            .with_columns([
                pl.col("reco_date").cast(pl.Date),
                pl.col("symbol").str.to_uppercase().str.strip_chars(),
                (t - pl.col("reco_date")).dt.days().alias("reco_age"),
            ])
            .filter(
                (pl.col("reco_date") < t) &
                (pl.col("reco_age") >= 0) &
                (pl.col("reco_age") <= max_days_valid)
            )
            .sort(["analyst_id", "symbol", "reco_date"])
            .unique(subset=["analyst_id", "symbol"], keep="last")
        )

    def _compute_imp_returns(self, df_tp: pl.DataFrame) -> pl.DataFrame:
        """
        Computed implied forcasted returns as :
            R_i_t = TP_i_t / P_i_t - 1
        Please note that the implied return is built with current prices at t.

        Parameters
        ----------
        df_tp: polars.DataFrame
            DataFrame with valid target prices informations and market prices
        
        Returns
        -------
        polars.DataFrame
            DataFrame containing the implied forcasted returns
        """
        return (
            df_tp
            .with_columns([
                (pl.col("target_price") / pl.col("price_at_pub") - 1.0).alias("imp_return_pub"),
                (pl.col("target_price") / pl.col("price_at_rebal") - 1.0).alias("imp_return"),
            ])
        )

    def _apply_decay(self, df_tp: pl.DataFrame) -> pl.DataFrame:
        """
        Perform a decay on timplied forcasted returns to lower the conviction
        on old target prices. The implied weights will be smaller in absolute value.

        Parameters
        ----------
        df_tp: polars.DataFrame
            DataFrame with implied forcasted returns

        Returns
        -------
        polars.DataFrame
            DataFrame containing the decayed implied forcasted returns
        """
        half_life_days = self.decay_half_life * 30.4
        return df_tp.with_columns(
            (pl.col("imp_return") * (0.5 ** (pl.col("reco_age") / half_life_days))).alias("decayed_imp_return")
        )
        
    def _compute_weights(self, df_tp: pl.DataFrame) -> pl.DataFrame:
        """
        Compute the implied weights for each stock in analyst coverage :
            - w_k+ = r_imp_k / sum(r_imp+)
            - w_k- = - r_imp_k / - sum(r_imp-)
        It results a weight neutral long-short portfolio.

        Parameters
        ----------
        df_tp: polars.DataFrame
            DataFrame with decayed implied forcasted returns

        Returns
        -------
        polars.DataFrame
            DataFrame containing the implied weights
        """
        base = (
            pl
            .when(pl.col("decayed_imp_return").is_not_null())
            .then(pl.col("decayed_imp_return"))
            .otherwise(pl.col("imp_return"))
        )

        return (
            df_tp
            .with_columns([
                base.alias("r_hat"),
                pl.when(base > 0).then(base).otherwise(0.0).alias("pos"),
                pl.when(base < 0).then(-base).otherwise(0.0).alias("neg"),
            ])
            .with_columns([
                pl.col("pos").sum().over("analyst_id").alias("sum_pos"),
                pl.col("neg").sum().over("analyst_id").alias("sum_neg"),
            ])
            .with_columns([
                pl.when(pl.col("r_hat") > 0)
                  .then(pl.when(pl.col("sum_pos") > 0)
                        .then(pl.col("r_hat")/pl.col("sum_pos"))
                        .otherwise(0.0))
                  .otherwise(
                      -pl.when(pl.col("sum_neg") > 0)
                        .then((-pl.col("r_hat"))/pl.col("sum_neg"))
                        .otherwise(0.0)
                  ).alias("w"),
                pl.when(pl.col("r_hat") > 0).then(pl.lit("L")).otherwise(pl.lit("S")).alias("side"),
            ])
            .with_columns(pl.col("w").abs().alias("abs_w"))
            .drop(["pos","neg","sum_pos","sum_neg"])
        )
