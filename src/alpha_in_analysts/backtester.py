import polars as pl
import logging
from datetime import date

from .book_engine import BookEngine
from .features_engine import FeaturesEngine
from .model_wrappers.model_wrapper import ModelWrapper
from .meta_portfolio import MetaPortfolio
from .utils.config import Config
from .utils.mapping import ModelType
from .utils.helpers import _parse_date, _add_months, _month_range, _last_trading_day_of_month
from .utils.s3_utils import s3Utils

logger = logging.getLogger(__name__)


class Backtester:
    """
    High level orchestrator dedicated to the backtesting process of the strategy based on machine learning models.

    It uses high level components to perform the following tasks:
        - BookEngine: Process the analysts' target prices to generate an implied portfolio.
        - FeatureEngine: Process analysts' books to generate features for the models.
        - ModelWrapper(s): Load trained machine learning models to perform inference on the generated features. 
        - MetaPortfolio: Construct and evaluate the strategy portfolio based on model predictions with a given methodology.

    Note: The models are assumed to be already trained and saved on disk by a separate component (Orchestrator).
          This part of training and saving is not handled here and is considered abstracted via the wrapper interface.
          Whatever model is used, it should be loadable and usable via the ModelWrapper interface.

    The backtesting is done as follows:
        0. Data is available from 2000 to 2025 - split into train (2000-2020) and test (2020-2025).
        1. A warm-up period is defined (e.g., 2018-2020) to allow feature generation prior to backtesting.
        2. For each month of the test period (2020-2025):
            a. Generate books from available analysts' target prices up to that month (x months of validity).
            b. Generate features from the generated books (current month and historical data).
            c. Load the trained model from disk.
            d. Perform inference on the generated features to obtain predictions.
            e. Construct the strategy portfolio based on model predictions using a defined methodology.
        3. Analyze the performance of the strategy over the backtest period using various metrics and visualizations.
    """

    def __init__(self, config: Config):
        """
        Parameters
        ----------
        config : Config
            Configuration object holding settings for the backtesting process.
        """
        self.config = config
        self.model: ModelWrapper = None
        self.ptf: MetaPortfolio = MetaPortfolio(dead_zone=self.config.transfo_dead_zone,
                                                k_pos=self.config.transfo_k_pos,
                                                k_neg=self.config.transfo_k_neg)
        self._load_model()

    def run(self):
        """
        Run the backtesting process as described in the class docstring.
        """
        logger.info("Starting backtesting process...")
        logger.info(f"Start Date: {self.config.test_start_date}, End Date: {self.config.test_end_date}")

        timeline = self._build_timeline()

        df_tp, df_prices = self._load_data(start_date=_add_months(timeline[0], -1), end_date=timeline[-1])

        weight_history: list[pl.DataFrame] = []

        features_engine = FeaturesEngine(df_target_prices=df_tp,
                                        df_prices=df_prices,
                                        validity_length=self.config.validity_length,
                                        decay_half_life=self.config.decay_halflife)
        
        res = features_engine.build_all_features(up_to_date=timeline[-1],
                                                 lookback_perf_pct=12,
                                                 lookback_perf=6,
                                                 lookback_vol_pct=6,
                                                 lookback_vol=6,
                                                 lookback_mean_ret=6,
                                                 drop_na=True)

        for t in timeline:
            logger.info(f"\tRunning backtest for date: {t}")

            book_t = BookEngine(df_tp=df_tp,
                                df_prices=df_prices,
                                validity_length=self.config.validity_length,
                                decay_half_life=self.config.decay_halflife
                                ).at_snapshot(snapshot_date=t)

            X_t: pl.DataFrame = res.filter(pl.col("date") == t).drop("date")
            y_hat_t = (
                X_t
                .select("analyst_id")
                .with_columns(pl.Series("predicted_pnl", self.model.predict(X=X_t)))
            )

            w_t = self.ptf.create_metaportfolio(
                analyst_books=book_t,
                predict_pnl=y_hat_t,
                method=self.config.construction_method,
                normalize=self.config.normalize_weights
            )
            w_t = w_t.with_columns(pl.lit(t).alias("date"))
            weight_history.append(w_t)

        logger.info("Backtesting process completed, analyzing results...")

        self._analyst_backtest(ptf_weights=weight_history, prices=df_prices, timeline=timeline)

    # -----------------------------------------------------------------
    # |                       Private Helpers                         |
    # -----------------------------------------------------------------

    def _load_model(self):
        """
        Load the trained model from disk using the ModelWrapper interface.
        """
        try:
            self.model = ModelType[self.config.model_name].value()
            self.model.load_model(model_path=self.config.model_paths.get('SKOPS'),
                                  manifest_path=self.config.model_paths.get('MANIFEST'))
            logger.info(f"Model {self.config.model_name} loaded successfully from disk.")
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
        
    def _load_data(self, start_date: date = None, end_date: date = None) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Load data from S3 using paths defined in the configuration.

        Parameters
        ----------
        start_date : date, optional
            Start date to filter data for backtesting, by default None
        end_date : date, optional
            End date to filter data for backtesting, by default None

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame]
            A tuple containing the target prices DataFrame and prices DataFrame.
        """
        df_tp = s3Utils.pull_parquet_file_from_s3(path=self.config.target_price_path, to_polars=True)
        df_prices = s3Utils.pull_parquet_file_from_s3(path=self.config.prices_path, to_polars=True)

        if start_date and end_date:
            df_tp = df_tp.filter((pl.col("reco_date") >= start_date) & (pl.col("reco_date") <= end_date))
            df_prices = df_prices.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))

        logger.info(f"Data loaded from S3: Target Prices ({df_tp.shape}), Prices ({df_prices.shape})")
        return df_tp, df_prices
    
    def _build_timeline(self) -> list[date]:
        """
        Build the backtesting timeline based on the configuration.
        Corresponds to a list of dates from test_start_date to test_end_date.
            
        Returns
        -------
        list[date]
            A list of dates for the backtest period.
        """
        start_date = _parse_date(self.config.test_start_date)
        end_date = _parse_date(self.config.test_end_date)

        backtest = _month_range(start_date, end_date)
        
        logger.info(f"Backtesting timeline constructed:{len(backtest)} months")

        return backtest
    
    def _compute_monthly_returns(self, df_prices: pl.DataFrame, month_ends: list[date]) -> pl.DataFrame:
        """
        Compute monthly returns for each stock based on end-of-month prices.
        Returns with a given date M (yyyy-mm-dd) correspond to the return from date M to date M+1.

        Ex : 
            Return for 2020-01-31 is computed as (Price at 2020-02-28 / Price at 2020-01-31) - 1

        Parameters
        ----------
        df_prices : pl.DataFrame
            DataFrame containing price data with columns ['date', 'stock_id', 'price'].
        month_ends : list[date]
            List of end-of-month dates to compute returns for.

        Returns
        -------
        pl.DataFrame
            DataFrame containing monthly returns with columns ['date', 'stock_id', 'ret'].
        """
        month_ends_df = (
            pl.DataFrame({"date": month_ends})
            .with_columns(pl.col("date").cast(pl.Date))
            .sort("date")
        )

        prices = (
            df_prices
            .select(["date", "stock_id", "price"])
            .with_columns(pl.col("date").cast(pl.Date))
            .sort(["stock_id", "date"])
        )

        stocks = prices.select("stock_id").unique()

        grid = stocks.join(month_ends_df, how="cross").sort(["stock_id", "date"])

        eom_prices = (
            grid
            .join_asof(prices, on="date", by="stock_id", strategy="backward")
            .rename({"price": "p_open"})
            .sort(["stock_id", "date"])
        )

        return (
            eom_prices
            .with_columns(pl.col("p_open").shift(-1).over("stock_id").alias("p_next"))
            .with_columns(((pl.col("p_next") / pl.col("p_open")) - 1.0).alias("ret"))
            .drop_nulls(["p_open", "p_next", "ret"])
            .select(["date", "stock_id", "ret"])
            .sort(["date", "stock_id"])
            .cast({"date": pl.Date})
        )
    
    def _analyst_backtest(self, ptf_weights: list[pl.DataFrame], prices: pl.DataFrame, timeline: list[date]) -> pl.DataFrame:
        """
        Analyse the backtest performance of the meta-analyst portfolio strategy.
        Save the performance results to an Excel file.

        Parameters
        ----------
        ptf_weights : list[pl.DataFrame]
            List of DataFrames containing portfolio weights for each backtest date.
        prices : pl.DataFrame
            DataFrame containing price data for the assets.
        timeline : list[date]
            List of dates corresponding to the backtest timeline.

        Returns
        -------
        pl.DataFrame
            DataFrame containing the backtest performance results.
        """
        df_weights = pl.concat(ptf_weights).select(["date", "stock_id", "meta_weight"]).cast({"date": pl.Date})

        df_returns = self._compute_monthly_returns(df_prices=prices, month_ends=timeline)

        perf = (
            df_weights
            .join(df_returns, on=["date", "stock_id"], how="inner")
            .with_columns(pl.col("ret").fill_null(0.0))
            .with_columns((pl.col("meta_weight") * pl.col("ret")).alias("weighted_ret"))
            .group_by("date")
            .agg(pl.col("weighted_ret").sum().alias("strategy_ret"))
            .sort("date")
            .with_columns([(pl.col("strategy_ret") + 1).cum_prod().alias("cumulative_ret")])
        )
        logger.info("Backtest performance analysis completed.")

        # path = self.config.backtest_base_path + f"/backtest_{self.config.model_name}_{self.config.test_start_date}_{self.config.test_end_date}_{date.today()}.xlsx"
        path2 = "outputs" + f"/backtest_{self.config.model_name}_{self.config.test_start_date}_{self.config.test_end_date}_{date.today()}.xlsx"
        # perf.write_excel(path)
        perf.write_excel(path2)
        # logger.info(f"Backtest performance saved to {path}")

        return perf