import polars as pl
import pandas_market_calendars as mcal
from src.alpha_in_analysts.book_engine import BookEngine
from src.alpha_in_analysts.backtester.portfolio import CreatePortfolio
from src.alpha_in_analysts.backtester.backtester import Backtest
from src.alpha_in_analysts.utils.s3_utils import s3Utils

class FeaturesEngine(BookEngine):
    """
    High-level module responsible for building analysts' features.
    Those features are used to predict analysts' future realised PnLs.

    Predicted future PnLs or implied PnLs will then be use as a reliability metric
    for analysts. Analysts that have a high predicted PnLs are considered to be a
    source of information for which we want to be long. Their implied portfolios
    will have a positive weight in the global meta-portfolio of analysts.
    In the contrary, analysts with a negative predicted PnLs will be shorted.

    Engineered features are of different kinds:
        - Convictions
        - Reco age and revisions
        - Coverage
        - Turnover
        - Track-record
        - Comparison / ranking
    """

    def __init__(self,
                 df_prices: pl.DataFrame,
                 df_tp: pl.DataFrame,
                 validity_length: int = 12,
                 decay_half_life: int = 6,
                 start_date: str = "1999-01-01",
                 end_date: str = "2024-12-31"
                 ):
        """
        Parameters
        ----------
        df_prices : pl.DataFrame
            Prices dataframe.
        df_tp : pl.DataFrame
            Target prices dataframe.
        validity_length : int
            Validity length for target prices (in months).
        decay_half_life : int
            Decay half-life for target prices (in months).
        start_date : str
            Start date for the features engine.
        end_date : str
            End date for the features engine.
        """
        super().__init__(
            df_tp=df_tp,
            df_prices=df_prices,
            validity_length=validity_length,
            decay_half_life=decay_half_life
        )
        self.start_date = pl.Series([start_date]).str.strptime(pl.Date).item()
        self.end_date = pl.Series([end_date]).str.strptime(pl.Date).item()
        self.df_books = None
        self.df_books_wide = None
        self.pnl_all_analysts = None
        self.coverage = None
        self.y = None
        self.dates = None

    def _build_books(self,
                     return_bool: bool = False
                     ):
        """
        Build books (collection of book at each date) dataframe in long format.
        Parameters
        ----------
        return_bool : bool
            Whether to return the df or just store it.
        """
        dates_range = self.df_prices.select(pl.col("date")).unique().sort(by="date").to_series().to_list()
        engine = BookEngine(df_tp=self.df_tp, df_prices=self.df_prices)

        for i, date in enumerate(dates_range):
            date = date.strftime("%Y-%m-%d")
            df_book_tmp = engine.at_snapshot(snapshot_date=date)

            df_book_tmp = df_book_tmp.with_columns(
                pl.lit(date).str.strptime(pl.Date, format="%Y-%m-%d").alias("date")
            )

            if i == 0:
                df_book = df_book_tmp
            else:
                df_book = pl.concat([df_book, df_book_tmp], how="vertical")

        if self.df_books is None:
            self.df_books = df_book
        if return_bool:
            return df_book

    def _build_books_wide(self,
                          return_bool: bool = False
                          ):
        """
        Build books dataframe in wide format.
        Parameters
        ----------
        return_bool : bool
            Whether to return the df or just store it.
        """
        if self.df_books is None:
            self._build_books(return_bool=False)

        df_book_wide = (
            self.df_books
            .pivot(
                values="weight",
                index="date",
                on=["analyst_id", "stock_id"]
            )
        )
        cols = df_book_wide.columns
        sorted_cols = (
                ["date"]
                + sorted(c for c in cols if c != "date")
        )

        if self.df_books_wide is None:
            self.df_books_wide = df_book_wide.select(sorted_cols)
        if return_bool:
            return df_book_wide.select(sorted_cols)

    def _build_weights_per_analyst(self,
                                   analyst_id: int
                                   ):
        """
        Build weights df for a given analyst.
        Parameters
        ----------
        analyst_id : int
            Analyst identifier.
        """
        df_w = (
            self.df_books
            .filter(pl.col("analyst_id") == analyst_id)
            .select(["weight", "stock_id", "date"])
        )

        stock_cols_ana = (
            df_w
            .select("stock_id")
            .unique()
            .sort("stock_id")
            .to_series()
            .cast(pl.Utf8)
            .to_list()
        )

        df_weights_pivoted = (
            df_w
            .pivot(
                values="weight",
                index="date",
                on="stock_id"
            )
            .select(["date"] + stock_cols_ana)
            .sort("date")
        )

        # NYSE trading calendar
        nyse = mcal.get_calendar("NYSE")
        # Do the union of nyse, amex and nasdaq

        schedule = nyse.schedule(
            start_date=self.start_date,
            end_date=self.end_date
        )

        trading_days = (
            pl.from_pandas(
                schedule.index.to_frame(name="date")
            )
            .with_columns(pl.col("date").cast(pl.Date))
        )

        month_end_trading_days = (
            trading_days
            .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
            .group_by("month")
            .agg(pl.col("date").max().alias("date"))
            .select("date")
            .sort("date")
        )

        df_weights_pivoted_filled = (
            month_end_trading_days
            .join(df_weights_pivoted, on="date", how="left")
            .sort("date")
            # .with_columns(
            #     pl.exclude("date").fill_null(strategy="forward", limit=2)
            # )
        )

        return {
            "df_weights_pivoted":df_weights_pivoted_filled,
            "stock_cols_ana":stock_cols_ana
        }


    def _build_prices_per_analyst(self,
                                  stock_cols_ana: list[str]
                                  ):
        """
        Build prices df for a given analyst.
        Parameters
        ----------
        stock_cols_ana : list[str]
            List of stock columns relevant for the given analyst.
        """
        stock_cols = (
            self.df_prices
            .select("stock_id")
            .unique()
            .sort("stock_id")
            .to_series()
            .cast(pl.Utf8)
            .to_list()
        )

        df_prices_wide = (
            self.df_prices
            .with_columns(pl.col("date").cast(pl.Date))  # optional
            .pivot(
                values="price",
                index="date",
                on="stock_id"
            )
            .select(["date"] + stock_cols)
            .sort("date")
        )
        df_prices_wide = df_prices_wide.select(["date"] + stock_cols_ana)
        return df_prices_wide

    @staticmethod
    def _build_returns_per_analyst(df_prices_wide: pl.DataFrame,
                                   df_weights_pivoted: pl.DataFrame
                                   ):
        """
        Build returns df for a given analyst.
        Parameters
        ----------
        df_prices_wide : pl.DataFrame
            Prices dataframe in wide format for the given analyst.
        df_weights_pivoted : pl.DataFrame
            Weights dataframe in wide format for the given analyst.
        """
        df_returns = (
            df_prices_wide
            .sort("date")
            .with_columns([
                pl.col(col)
                .pct_change()
                .alias(col)
                for col in df_prices_wide.columns
                if col != "date"
            ])
        )

        df_returns = (
            df_returns
            .with_columns(pl.col("date").cast(pl.Date))
            .filter(
                pl.col("date").is_in(
                    pl.lit(df_weights_pivoted["date"].cast(pl.Date)).implode()
                )
            )
        )
        # !!! Shift returns by -1 to align with weights
        df_returns = df_returns.with_columns(
            pl.exclude("date").shift(-1)
        )

        return df_returns

    def _build_pnl_per_analyst(self,
                               analyst_id: int,
                               tc: int|float = 10,
                               strategy_name: str = "strat_ret"
                               ):
        """
        Build the PnL for a given analyst.
        Parameters
        ----------
        analyst_id : int
            Analyst identifier.
        tc : int|float
            Transaction costs (in bps).
        strategy_name : str
            Name of the strategy (used in backtester).
        """
        # Pre-requisites for backtesting i.e. computing the pnl
        wpa = self._build_weights_per_analyst(analyst_id=analyst_id)
        df_weights_pivoted = wpa["df_weights_pivoted"]
        df_prices_wide = self._build_prices_per_analyst(stock_cols_ana=wpa["stock_cols_ana"])
        df_returns = self._build_returns_per_analyst(
            df_prices_wide=df_prices_wide,
            df_weights_pivoted=df_weights_pivoted
        )

        # Compute coverage i.e. non-null weights across assets at each date
        df_coverage = df_weights_pivoted.select([
            "date",
            pl.sum_horizontal(pl.all().exclude("date").is_not_null()).alias("coverage")
        ])
        df_coverage = df_coverage.with_columns(
            pl.lit(analyst_id).alias("analyst_id")
        )

        ptf = CreatePortfolio(
            returns=df_returns,
            weights=df_weights_pivoted,
            rebal_periods=1
        )
        ptf.rebalance_portfolio_irregular()

        backtester = Backtest(
            returns=df_returns,
            weights=ptf.rebalanced_weights,
            turnover=ptf.turnover,
            transaction_costs=tc,
            strategy_name=strategy_name
        )
        backtester.run_backtest()

        strat_ret = backtester.portfolio_net_returns
        strat_ret = strat_ret.with_columns(
            pl.lit(analyst_id).alias("analyst_id")
        )
        return {
            "strat_ret": strat_ret,
            "df_coverage": df_coverage
        }

    def _build_pnl_all_analysts(self,
                                tc: int|float = 10,
                                strategy_name: str = "strat_ret",
                                ret_bool: bool = False
                                ):
        """
        Build the PnL for all analysts.
        Parameters
        ----------
        tc : int|float
            Transaction costs (in bps).
        strategy_name : str
            Name of the strategy (used in backtester).
        ret_bool : bool
            Whether to return the df or just store it.
        """
        if self.df_books is None:
            self._build_books(return_bool=False)

        analyst_ids = (
            self.df_books
            .select("analyst_id")
            .unique()
            .sort("analyst_id")
            .to_series()
            .to_list()
        )

        # random sample of analysts for testing
        import pandas as pd
        a = pd.Series(analyst_ids).sample(15, random_state=42).tolist()
        for i, analyst_id in enumerate(a):
            print(f"Building PnL for analyst {analyst_id} ({i+1}/{len(analyst_ids)})")
            res = self._build_pnl_per_analyst(
                analyst_id=analyst_id,
                tc=tc,
                strategy_name=strategy_name
            )
            strat_ret_tmp = res["strat_ret"]
            df_coverage_tmp = res["df_coverage"]

            if i == 0:
                strat_ret = strat_ret_tmp
                coverage = df_coverage_tmp
            else:
                strat_ret = pl.concat([strat_ret, strat_ret_tmp], how="vertical")
                coverage = pl.concat([coverage, df_coverage_tmp], how="vertical")

        if self.pnl_all_analysts is None:
            self.pnl_all_analysts = strat_ret
        if self.coverage is None:
            self.coverage = coverage
        if ret_bool:
            return strat_ret

    def _build_cum_perf(self, up_to_date:str, n:int=6, pct:bool=False):
        """
        Cumulative performance over the last n months.
        Parameters
        ----------
        up_to_date : str
            Date at which to compute the feature (formation date).
        n : int
            Lookback period (in months) for cumulative performance feature.
        pct : bool
            Whether to return the percentile rank of the cumulative performance.
        """
        # Convert the up_to_date to pl.Date
        up_to_date_pl = pl.Series([up_to_date]).str.strptime(pl.Date).item()

        # Keep last 6 months up to formation date
        window = (
            self.pnl_all_analysts
            .filter(pl.col("date") <= up_to_date_pl)
            .sort(["analyst_id", "date"])
            .group_by("analyst_id")
            .tail(n)
        )

        perf_n_m = (
            window
            .group_by("analyst_id")
            .agg([
                # number of non-null returns
                pl.col("strat_ret").count().alias("n_obs"),

                # total number of rows in the window
                pl.len().alias("n_expected"),

                # geometric performance
                ((1 + pl.col("strat_ret")).product() - 1).alias(f"perf_{n}m"),
            ])
            .with_columns(
                pl.when(pl.col("n_obs") == pl.col("n_expected"))
                .then(pl.col(f"perf_{n}m"))
                .otherwise(None)
                .alias(f"perf_{n}m")
            )
            .select(["analyst_id", f"perf_{n}m"])
        )

        if not pct:
            return perf_n_m

        # Cross-sectional percentile rank
        perf_pct_nm = perf_n_m.with_columns(
            (
                    (pl.col(f"perf_{n}m").rank(method="average") - 1)
                    / (pl.col(f"perf_{n}m").count() - 1)
            ).alias(f"perf_{n}m_pct")
        )

        return perf_pct_nm

    def _build_volatility(self, up_to_date:str, n:int=6, pct:bool=False):
        """
        Volatility over the last n months.
        Parameters
        ----------
        up_to_date : str
            Date at which to compute the feature (formation date).
        n : int
            Lookback period (in months) for volatility feature.
        pct : bool
            Whether to return the percentile rank of the volatility.
        """
        up_to_date_pl = pl.Series([up_to_date]).str.strptime(pl.Date).item()

        window = (
            self.pnl_all_analysts
            .filter(pl.col("date") <= up_to_date_pl)
            .sort(["analyst_id", "date"])
            .group_by("analyst_id")
            .tail(n)
        )

        vol_nm = (
            window
            .group_by("analyst_id")
            .agg([
                pl.col("strat_ret").count().alias("n_obs"),
                pl.len().alias("n_expected"),
                pl.col("strat_ret").std().alias(f"vol_{n}m"),
            ])
            .with_columns(
                pl.when(pl.col("n_obs") == pl.col("n_expected"))
                .then(pl.col(f"vol_{n}m"))
                .otherwise(None)
                .alias(f"vol_{n}m")
            )
            .select(["analyst_id", f"vol_{n}m"])
        )

        if not pct:
            return vol_nm

        vol_pct_nm = vol_nm.with_columns(
            (
                    (pl.col(f"vol_{n}m").rank(method="average") - 1)
                    / (pl.col(f"vol_{n}m").count() - 1)
            ).alias(f"vol_{n}m_pct")
        )
        return vol_pct_nm

    def _build_mean_ret(self, up_to_date:str, n:int=6, pct:bool=False):
        """
        Mean return over the last n months.
        Parameters
        ----------
        up_to_date : str
            Date at which to compute the feature (formation date).
        n : int
            Lookback period (in months) for mean return feature.
        pct : bool
            Whether to return the percentile rank of the mean return.
        """
        up_to_date_pl = pl.Series([up_to_date]).str.strptime(pl.Date).item()

        window = (
            self.pnl_all_analysts
            .filter(pl.col("date") <= up_to_date_pl)
            .sort(["analyst_id", "date"])
            .group_by("analyst_id")
            .tail(n)
        )

        mean_ret_nm = (
            window
            .group_by("analyst_id")
            .agg([
                pl.col("strat_ret").count().alias("n_obs"),
                pl.len().alias("n_expected"),
                pl.col("strat_ret").mean().alias(f"mean_ret_{n}m"),
            ])
            .with_columns(
                pl.when(pl.col("n_obs") == pl.col("n_expected"))
                .then(pl.col(f"mean_ret_{n}m"))
                .otherwise(None)
                .alias(f"mean_ret_{n}m")
            )
            .select(["analyst_id", f"mean_ret_{n}m"])
        )

        if not pct:
            return mean_ret_nm

        mean_ret_pct_nm = mean_ret_nm.with_columns(
            (
                    (pl.col(f"mean_ret_{n}m").rank(method="average") - 1)
                    / (pl.col(f"mean_ret_{n}m").count() - 1)
            ).alias(f"mean_ret_{n}m_pct")
        )

        return mean_ret_pct_nm

    def _build_sharpe(self, up_to_date: str, n: int = 12, pct: bool = False):
        """
        Annualized Sharpe ratio over the last n months (rf = 0).

        Parameters
        ----------
        up_to_date : str
            Date at which to compute the feature (formation date).
        n : int
            Lookback period (in months) for Sharpe ratio.
        pct : bool
            Whether to return the percentile rank of the Sharpe ratio.
        """
        up_to_date_pl = pl.Series([up_to_date]).str.strptime(pl.Date).item()

        window = (
            self.pnl_all_analysts
            .filter(pl.col("date") <= up_to_date_pl)
            .sort(["analyst_id", "date"])
            .group_by("analyst_id")
            .tail(n)
        )

        sharpe_nm = (
            window
            .group_by("analyst_id")
            .agg([
                pl.col("strat_ret").count().alias("n_obs"),
                pl.len().alias("n_expected"),
                pl.col("strat_ret").mean().alias("mean_ret"),
                pl.col("strat_ret").std().alias("std_ret"),
            ])
            .with_columns(
                (
                    pl.when(
                        (pl.col("n_obs") == pl.col("n_expected"))
                        & (pl.col("std_ret") > 0)
                    )
                    .then((pl.lit(12.0).sqrt()) * pl.col("mean_ret") / pl.col("std_ret"))
                    .otherwise(None)
                ).alias(f"sharpe_{n}m")
            )
            .select(["analyst_id", f"sharpe_{n}m"])
        )

        if not pct:
            return sharpe_nm

        sharpe_pct_nm = sharpe_nm.with_columns(
            (
                    (pl.col(f"sharpe_{n}m").rank(method="average") - 1)
                    / (pl.col(f"sharpe_{n}m").count() - 1)
            ).alias(f"sharpe_{n}m_pct")
        )

        return sharpe_pct_nm

    def _build_sortino(self, up_to_date: str, n: int = 6, pct: bool = False):
        """
        Annualized Sortino ratio over the last n months (rf = 0).

        Parameters
        ----------
        up_to_date : str
            Date at which to compute the feature (formation date).
        n : int
            Lookback period (in months).
        pct : bool
            Whether to return the percentile rank of the Sortino ratio.
        """
        up_to_date_pl = pl.Series([up_to_date]).str.strptime(pl.Date).item()

        window = (
            self.pnl_all_analysts
            .filter(pl.col("date") <= up_to_date_pl)
            .sort(["analyst_id", "date"])
            .group_by("analyst_id")
            .tail(n)
        )

        sortino_nm = (
            window
            .group_by("analyst_id")
            .agg([
                pl.col("strat_ret").count().alias("n_obs"),
                pl.len().alias("n_expected"),
                pl.col("strat_ret").mean().alias("mean_ret"),
                pl.col("strat_ret")
                .filter(pl.col("strat_ret") < 0)
                .std()
                .alias("downside_std"),
            ])
            .with_columns(
                (
                    pl.when(
                        (pl.col("n_obs") == pl.col("n_expected")) &
                        (pl.col("downside_std") > 0)
                    )
                    .then(
                        pl.col("mean_ret") / pl.col("downside_std") * (12 ** 0.5)
                    )
                    .otherwise(None)
                ).alias(f"sortino_{n}m")
            )
            .select(["analyst_id", f"sortino_{n}m"])
        )

        if not pct:
            return sortino_nm

        sortino_pct_nm = sortino_nm.with_columns(
            (
                    (pl.col(f"sortino_{n}m").rank(method="average") - 1)
                    / (pl.col(f"sortino_{n}m").count() - 1)
            ).alias(f"sortino_{n}m_pct")
        )

        return sortino_pct_nm

    def _build_mean_coverage(self, up_to_date:str, n:int=6, pct:bool=False):
        """
        Mean coverage over the last n months.
        Parameters
        ----------
        up_to_date : str
            Date at which to compute the feature (formation date).
        n : int
            Lookback period (in months) for mean return feature.
        pct : bool
            Whether to return the percentile rank of the mean return.
        """
        up_to_date_pl = pl.Series([up_to_date]).str.strptime(pl.Date).item()

        window = (
            self.coverage
            .filter(pl.col("date") <= up_to_date_pl)
            .sort(["analyst_id", "date"])
            .group_by("analyst_id")
            .tail(n)
        )

        mean_coverage_nm = (
            window
            .group_by("analyst_id")
            .agg([
                pl.col("coverage").count().alias("n_obs"),
                pl.len().alias("n_expected"),
                pl.col("coverage").mean().alias(f"coverage_{n}m"),
            ])
            .with_columns(
                pl.when(pl.col("n_obs") == pl.col("n_expected"))
                .then(pl.col(f"coverage_{n}m"))
                .otherwise(None)
                .alias(f"coverage_{n}m")
            )
            .select(["analyst_id", f"coverage_{n}m"])
        )

        if not pct:
            return mean_coverage_nm

        mean_coverage_pct_nm = mean_coverage_nm.with_columns(
            (
                    (pl.col(f"coverage_{n}m").rank(method="average") - 1)
                    / (pl.col(f"coverage_{n}m").count() - 1)
            ).alias(f"coverage_{n}m_pct")
        )

        return mean_coverage_pct_nm

    def _build_all_features(self,
                           up_to_date:str,
                           lookback_perf_pct:int=12,
                           lookback_perf:int=6,
                           lookback_vol_pct:int=6,
                           lookback_vol: int = 6,
                           lookback_mean_ret:int=6,
                            lookback_recent_sharpe:int=6,
                            lookback_sharpe:int=12,
                            lookback_sortino:int=12,
                            lookback_recent_sortino:int=6
                           ):
        """
        Main method to build all features at once.
        Parameters
        ----------
        up_to_date : str
            Date at which to compute the features (formation date).
        lookback_perf_pct : int
            Lookback period (in months) for performance percentile feature.
        lookback_perf : int
            Lookback period (in months) for recent performance feature.
        lookback_vol_pct : int
            Lookback period (in months) for volatility percentile feature.
        lookback_vol : int
            Lookback period (in months) for recent volatility feature.
        lookback_mean_ret : int
            Lookback period (in months) for mean return feature.
        """
        if self.pnl_all_analysts is None:
            try:
                self.pnl_all_analysts = s3Utils.pull_parquet_file_from_s3(
                    path="s3://alpha-in-analysts-storage/data/pnl_all_analysts.parquet",
                    to_polars=True
                )
            except Exception as e:
                print("Could not load pnl_all_analysts from S3, building it locally.")
                self._build_pnl_all_analysts(ret_bool=False)

        # if self.coverage is None:
        #     try:
        #         self.coverage = s3Utils.pull_parquet_file_from_s3(
        #             path="s3://alpha-in-analysts-storage/data/coverage.parquet",
        #             to_polars=True
        #         )
        #     except Exception as e:
        #         print("Could not load coverage from S3, building it locally.")
        #         self._build_pnl_all_analysts(ret_bool=False)

        perf_pct = self._build_cum_perf(up_to_date=up_to_date,
                                        n=lookback_perf_pct,
                                        pct=True)
        perf = self._build_cum_perf(up_to_date=up_to_date,
                                        n=lookback_perf_pct,
                                        pct=False)
        recent_perf_pct = self._build_cum_perf(up_to_date=up_to_date,
                                               n=lookback_perf,
                                               pct=True)
        recent_perf = self._build_cum_perf(up_to_date=up_to_date,
                                               n=lookback_perf,
                                               pct=False)
        recent_vol_pct = self._build_volatility(up_to_date=up_to_date,
                                                n=lookback_vol_pct,
                                                pct=True)
        recent_vol = self._build_volatility(up_to_date=up_to_date,
                                            n=lookback_vol,
                                            pct=False)
        mean_ret = self._build_mean_ret(up_to_date=up_to_date,
                                        n=lookback_mean_ret,
                                        pct=False)
        mean_ret_pct = self._build_mean_ret(up_to_date=up_to_date,
                                        n=lookback_mean_ret,
                                        pct=True)
        recent_sharpe = self._build_sharpe(up_to_date=up_to_date,
                                    n=lookback_recent_sharpe,
                                    pct=False)
        recent_sharpe_pct = self._build_sharpe(up_to_date=up_to_date,
                                           n=lookback_recent_sharpe,
                                           pct=True)
        sharpe = self._build_sharpe(up_to_date=up_to_date,
                                           n=lookback_sharpe,
                                           pct=False)
        sharpe_pct = self._build_sharpe(up_to_date=up_to_date,
                                    n=lookback_sharpe,
                                    pct=True)
        recent_sortino = self._build_sortino(up_to_date=up_to_date,
                                    n=lookback_recent_sortino,
                                    pct=False)
        recent_sortino_pct = self._build_sortino(up_to_date=up_to_date,
                                           n=lookback_recent_sortino,
                                           pct=True)
        sortino = self._build_sortino(up_to_date=up_to_date,
                                             n=lookback_sortino,
                                             pct=False)
        sortino_pct = self._build_sortino(up_to_date=up_to_date,
                                    n=lookback_sortino,
                                    pct=True)

        # Add a column with the up_to_date date in each df, named 'date'
        features = [perf_pct, recent_perf_pct, recent_vol_pct, recent_vol, mean_ret,
                    mean_ret_pct, perf, recent_perf, recent_sharpe, recent_sharpe_pct,
                    sharpe, sharpe_pct, recent_sortino, recent_sortino_pct, sortino, sortino_pct]
        dates = self.df_prices.select(pl.col("date")).unique().sort("date").to_series().to_list()
        date_x_index = dates.index(pl.Series([up_to_date]).str.strptime(pl.Date).item())
        date_y = str(dates[date_x_index+12])
        features = [
            df
            .with_columns(
                pl.lit(date_y).str.strptime(pl.Date).alias("date")
            )
            .select(
                ["date"] + [c for c in df.columns if c != "date"]
            )
            for df in features
        ]

        # Finally, horizontally concatenate all features into a single df
        df_features = features[0]

        for df in features[1:]:
            df_features = df_features.join(
                df,
                on=["date", "analyst_id"],
                how="inner"
            )
        df_features = df_features.drop_nulls()
        return df_features

    def _build_y(self, up_to_date:str=None, lookback:int=12):

        y = self._build_cum_perf(up_to_date=up_to_date,
                                    n=lookback,
                                    pct=False)
        y = (
            y
            .with_columns(
                pl.lit(up_to_date).str.strptime(pl.Date).alias("date")
            )
            .select(
                ["date"] + [c for c in y.columns if c != "date"]
            )
        )
        y = y.rename({"perf_12m": "y"})
        self.y = y

    def get_features_and_y(self,
                           up_to_date:str,
                           lookback_perf_pct:int=12,
                           lookback_perf:int=6,
                           lookback_vol_pct:int=6,
                           lookback_vol: int = 6,
                           lookback_mean_ret:int=6,
                            lookback_sharpe:int=12,
                            lookback_sortino:int=12,
                            lookback_recent_sharpe:int=6,
                            lookback_recent_sortino:int=6,
                           lookback_y:int=12
                           ):
        """
        Get features and target variable y at once.
        Parameters
        ----------
        up_to_date : str
            Date at which to compute the features (formation date).
        lookback_perf_pct : int
            Lookback period (in months) for performance percentile feature.
        lookback_perf : int
            Lookback period (in months) for recent performance feature.
        lookback_vol_pct : int
            Lookback period (in months) for volatility percentile feature.
        lookback_vol : int
            Lookback period (in months) for recent volatility feature.
        lookback_mean_ret : int
            Lookback period (in months) for mean return feature.
        lookback_sharpe : int
            Lookback period (in months) for Sharpe ratio feature.
        lookback_recent_sharpe : int
            Lookback period (in months) for recent Sharpe ratio feature.
        lookback_sortino : int
            Lookback period (in months) for Sortino ratio feature.
        lookback_recent_sortino : int
            Lookback period (in months) for recent Sortino ratio feature.
        lookback_y : int
            Lookback period (in months) for target variable y.
        Returns
        -------
        pl.DataFrame
            DataFrame with features and target variable y.
        """
        if self.dates is None:
            self.dates = (
                self.df_prices
                .select(pl.col("date").cast(pl.Date))
                .unique()
                .sort("date")
                .to_series()
                .to_list()
            )

        up_to_date_y_pl = pl.Series([up_to_date]).str.strptime(pl.Date).item()
        date_x_idx = self.dates.index(up_to_date_y_pl)
        if date_x_idx <= 12:
            raise ValueError("Not enough data to compute target variable y. Need at least 12 months + "
                             "max lookback period of the features of data before the up_to_date.")
        up_to_date_x = str(self.dates[date_x_idx-12])

        df_features = self._build_all_features(
            up_to_date=up_to_date_x,
            lookback_perf_pct=lookback_perf_pct,
            lookback_perf=lookback_perf,
            lookback_vol_pct=lookback_vol_pct,
            lookback_vol=lookback_vol,
            lookback_mean_ret=lookback_mean_ret,
            lookback_sharpe=lookback_sharpe,
            lookback_recent_sharpe=lookback_recent_sharpe,
            lookback_sortino=lookback_sortino,
            lookback_recent_sortino=lookback_recent_sortino
        )
        if self.y is None:
            self._build_y(up_to_date=str(up_to_date_y_pl), lookback=lookback_y)

        y_tmp = (
            self.y
            .select(["analyst_id", "y"])
        )
        df_final = (
            df_features
            .join(
                y_tmp,
                on="analyst_id",
                how="inner"
            )
        )
        df_final = df_final.drop_nulls()
        return df_final






