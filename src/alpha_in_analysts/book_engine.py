import polars as pl
from datetime import date, datetime, timedelta
from .utils.helpers import _validate_date
import logging

DateLike = date | str | datetime
logger = logging.getLogger(__name__)


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
            Needs columns :
                - analyst_id: Unique identifier of the analyst
                - stock_id: Unique identifier of the stock (permno)
                - reco_date: Date of the target price publication
                - target_price: Target price value
        df_prices: polars.DataFrame
            DataFrame with market prices informations
            Needs columns :
                - stock_id: Unique identifier of the stock (permno)
                - date: Date of the price
                - price: Market price value
        df_metadata: polars.DataFrame, default None
            DataFrame with stock metadata informations
            If not None, needs at least the column :
                - stock_id: Unique identifier of the stock (permno)
        validity_length: int, default 12
            Number of month for which a target price is considered still valid
        decay_half_life: int, default 6
            Parameter of the exponential decay, number of month needed to
            decrease the power of the recommandation by 2
        """
        if not {"analyst_id", "stock_id", "reco_date", "target_price"}.issubset(set(df_tp.columns)):
            raise ValueError("df_tp must contain columns : analyst_id, stock_id, reco_date, target_price")
        if not {"stock_id", "date", "price"}.issubset(set(df_prices.columns)):
            raise ValueError("df_prices must contain columns : stock_id, date, price")
        if df_metadata is not None and "stock_id" not in df_metadata.columns:
            raise ValueError("df_metadata must contain the column : stock_id")
        
        self.df_tp = (
            df_tp
            .select(["analyst_id", "stock_id", "reco_date", "target_price"])
            .with_columns(pl.col("reco_date").cast(pl.Date))
        )

        self.df_prices = (
            df_prices
            .select(["stock_id", "date", "price"])
            .with_columns(pl.col("date").cast(pl.Date))
        )
        self.df_metadata = df_metadata
        self.validity_length = validity_length
        self.decay_half_life = decay_half_life

    def at_snapshot(self, snapshot_date: DateLike, apply_decay: bool = True) -> pl.DataFrame:
        """
        Build analysts's books at a specific snapshot_date.

        Parameters
        ----------
        snapshort_date: date, datetime, pl.Date or string
            Date for which the engine will compute analysts's implied books
        apply_decay: bool, default True
            Whether to apply decay on implied forcasted returns

        Returns
        -------
        polars.DataFrame
            DataFrame containing the implied portfolios
        """
        logger.info("\t\tBuilding book at snapshot date: %s", str(snapshot_date))
        
        t = _validate_date(snapshot_date)
        df_tp_valid = self._filter_valid_tp(self.df_tp, t)

        df_full = self._join_prices(df_tp_valid, self.df_prices, t)
        
        df_imp_returns = self._compute_imp_returns(df_full)
        
        if apply_decay:
            df_decayed = self._apply_decay(df_imp_returns)
        else:
            df_decayed = df_imp_returns.with_columns(
                pl.lit(None).cast(pl.Float64).alias("decayed_imp_return")
            )

        df_weights = self._compute_weights(df_decayed)
        return (
            df_weights.join(self.df_metadata, on="symbol", how="left")
            if self.df_metadata is not None else df_weights
        )

    def _filter_valid_tp(self, df_tp: pl.DataFrame, t: date) -> pl.DataFrame:
        """
        Perform a temporal filter on target-prices (t - validity_length < t_tp < t).

        Parameters
        ----------
        df_tp: polars.DataFrame
            DataFrame with analysts's target prices informations
        t: datetime.date
            Snapshot date

        Returns
        -------
        polars.DataFrame
            DataFrame containing only valid target prices
        """
        max_days_valid = int(round(self.validity_length * 30.4))
        t_days = pl.lit(t).cast(pl.Date).cast(pl.Int32)
        reco_days = pl.col("reco_date").cast(pl.Date).cast(pl.Int32)

        return (
            df_tp
            .with_columns((t_days - reco_days).alias("reco_age"))
            .filter(
                (pl.col("reco_age") >= 0) &
                (pl.col("reco_age") <= max_days_valid)
            )
            .sort(["analyst_id", "stock_id", "reco_date"])
            .unique(subset=["analyst_id", "stock_id"], keep="last")
        )

    def _join_prices(
            self, 
            df_tp: pl.DataFrame, 
            df_prices: pl.DataFrame, 
            t: date,
            max_days_valid: int = 5
        ) -> pl.DataFrame:
        """
        Perform as-of join to get : 
            - price at recommendation date
            - price at snapshot date

        Parameters
        ----------
        df_tp: polars.DataFrame
            DataFrame with valid target prices informations
        df_prices: polars.DataFrame
            DataFrame with market prices informations
        t: datetime.date
            Snapshot date
        max_days_valid: int, default 5
            Maximum number of days between reco/rebal date and price date to consider a price valid
            Otherwise, the line will be dropped.

        Returns
        -------
        polars.DataFrame
            DataFrame containing target prices with market prices at reco and rebal dates
        """
        tol = timedelta(days=max_days_valid)
        df_prices_sorted = df_prices.sort(["stock_id", "date"])

        df_tp_with_reco = (
            df_tp
            .sort(["stock_id", "reco_date"])
            .join_asof(
                df_prices_sorted,
                left_on="reco_date",
                right_on="date",
                by="stock_id",
                strategy="backward",
                tolerance=tol,
                check_sortedness=False,
            )
            .rename({"price": "price_at_reco"})
            .drop("date")
        )

        df_rebal_keys = (
            df_tp.select("stock_id").unique()
            .with_columns(pl.lit(t).cast(pl.Date).alias("rebal_date"))
            .sort(["stock_id", "rebal_date"])
        )

        df_prices_rebal = (
            df_rebal_keys
            .join_asof(
                df_prices_sorted.rename({"date": "mkt_date"}),
                left_on="rebal_date",
                right_on="mkt_date",
                by="stock_id",
                strategy="backward",
                tolerance=tol,
                check_sortedness=False,
            )
            .select(["stock_id", pl.col("price").alias("price_at_rebal")])
        )

        return (
            df_tp_with_reco
            .join(df_prices_rebal, on="stock_id", how="left")
            .filter(pl.col("price_at_reco").is_not_null())
            .filter(pl.col("price_at_rebal").is_not_null())
        )
    
    def _compute_imp_returns(self, df_tp: pl.DataFrame) -> pl.DataFrame:
        """
        Computed implied forcasted returns as :
            R_i_t = TP_i_t / P_i_t - 1
        For safety, we filter out implied returns greater than 300% or lower than -300%.
        
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
                (pl.col("target_price") / pl.col("price_at_reco") - 1.0).alias("imp_return_reco"),
                (pl.col("target_price") / pl.col("price_at_rebal") - 1.0).alias("imp_return_rebal"),
            ])
            .filter((pl.col("imp_return_rebal") < 3.0) & (pl.col("imp_return_rebal") > -3.0))
            .filter((pl.col("imp_return_reco") < 3.0) & (pl.col("imp_return_reco") > -3.0))
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
            (pl.col("imp_return_rebal") * (0.5 ** (pl.col("reco_age") / half_life_days))).alias("decayed_imp_return")
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
            .otherwise(pl.col("imp_return_rebal"))
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
                  ).alias("weight"),
            ])
            .drop(["pos", "neg", "sum_pos", "sum_neg", "r_hat"])
        )
