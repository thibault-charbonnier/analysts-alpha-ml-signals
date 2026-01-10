import polars as pl
import logging
from datetime import date, datetime
import pandas as pd
from .book_engine import BookEngine
from .model_wrappers.model_wrapper import ModelWrapper
from .meta_portfolio import MetaPortfolio
from .utils.config import Config
from .utils.mapping import map_model
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
        self.ptf: MetaPortfolio = MetaPortfolio(dead_zone=self.config.transfo_dead_zone,
                                                k_pos=self.config.transfo_k_pos,
                                                k_neg=self.config.transfo_k_neg)

    def run(self, models: list[str]):
        """
        Run the backtesting process as described in the class docstring.

        Parameters
        ----------
        models : list[str]
            List of model names to backtest.
        """
        logger.info("Starting backtesting process...")
        logger.info(f"Start Date: {self.config.test_start_date}, End Date: {self.config.test_end_date}")
        logger.info("Models to backtest: " + ", ".join(models))

        timeline = self._build_timeline()

        df_tp, df_prices = self._load_data(start_date=_add_months(timeline[0], -1), end_date=timeline[-1])
        oos_pred = self._load_pred()

        weight_history: dict[str, list[pl.DataFrame]] = {model: [] for model in models}
        for t in timeline:
            logger.info(f"\tRunning backtest for date: {t}")

            book_t = BookEngine(df_tp=df_tp,
                                df_prices=df_prices,
                                validity_length=self.config.validity_length,
                                decay_half_life=self.config.decay_halflife
                                ).at_snapshot(snapshot_date=t)

            for model in models:
                if model == "BENCHMARK":
                    w_t = self._benchmark_from_book(book_t)
                elif model  == "EQUAL_WEIGHTED":
                    w_t = self.equal_weighted_analysts_ls(book_t)
                else:
                    y_hat_t = (
                        oos_pred
                        .filter(pl.col("date") == _add_months(t, -12))
                        .filter(pl.col("model") == map_model.get(model))
                        .with_columns((-pl.col("predicted_pnl")).alias("predicted_pnl"))
                    )
                    w_t = self.ptf.create_metaportfolio(
                        analyst_books=book_t,
                        predict_pnl=y_hat_t,
                        method=self.config.construction_method,
                        normalize=self.config.normalize_weights,
                        use_transfo=self.config.pnl_transfo
                    )

                w_t = self.force_50_50_ls(w_t, col="meta_weight")
                stats = self.exposure_stats(w_t)
                logger.info(f"\t\t[{model} @ {t}] gross={stats['gross']:.3f} net={stats['net']:.3f} "
                            f"long={stats['long']:.3f} short={stats['short']:.3f}")

                w_t = w_t.with_columns(pl.lit(t).alias("date"))
                weight_history[model].append(w_t)

        logger.info("Backtesting process completed, analyzing results...")

        history = {}
        for model in models:
            df_res = self._analyst_backtest(ptf_weights=weight_history[model], prices=df_prices, timeline=timeline, model=model)
            history[model] = df_res

        combined = None
        for model, df_res in history.items():
            pdf = df_res.select(["date", "cumulative_ret"]).to_pandas()
            pdf["date"] = pd.to_datetime(pdf["date"])
            pdf = pdf.set_index("date").sort_index()
            pdf = pdf.rename(columns={"cumulative_ret": model})
            combined = pdf if combined is None else combined.join(pdf, how="outer")

        out_path = f"outputs/backtest/backtest_cumulative_ret_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        with pd.ExcelWriter(out_path) as writer:
            combined.to_excel(writer, sheet_name="cumulative_ret")

        return combined
    # -----------------------------------------------------------------
    # |                       Private Helpers                         |
    # -----------------------------------------------------------------

    def force_50_50_ls(self, w: pl.DataFrame, col: str = "meta_weight") -> pl.DataFrame:
        long_sum = w.filter(pl.col(col) > 0).select(pl.col(col).sum()).item() or 0.0
        short_sum = w.filter(pl.col(col) < 0).select(pl.col(col).abs().sum()).item() or 0.0

        # si un côté est vide -> on ne peut pas forcer un LS propre
        if long_sum == 0.0 or short_sum == 0.0:
            return w  # ou raise / fallback benchmark

        return (
            w.with_columns(
                pl.when(pl.col(col) > 0)
                .then(pl.col(col) * (0.5 / pl.lit(long_sum)))
                .when(pl.col(col) < 0)
                .then(pl.col(col) * (0.5 / pl.lit(short_sum)))
                .otherwise(0.0)
                .alias(col)
            )
        )

    def exposure_stats(self, w: pl.DataFrame, col="meta_weight") -> dict:
        net_ = w.select(pl.col(col).sum()).item()
        gross_ = w.select(pl.col(col).abs().sum()).item()
        long_ = w.filter(pl.col(col) > 0).select(pl.col(col).sum()).item()
        short_ = -w.filter(pl.col(col) < 0).select(pl.col(col).sum()).item()
        return {"net": net_, "gross": gross_, "long": long_, "short": short_}

    def _benchmark_from_book(self, book_t: pl.DataFrame) -> pl.DataFrame:
        """
        Create benchmark meta-portfolio from the given analyst book by equally weighting stocks.
        """
        w = (
            book_t
            .group_by("stock_id")
            .agg(pl.col("weight").sum().alias("meta_weight"))
        )
        gross = w.select(pl.col("meta_weight").abs().sum()).item()
        if gross == 0:
            return w.with_columns(pl.lit(0.0).alias("meta_weight"))
        return w.with_columns((pl.col("meta_weight") / pl.lit(gross)).alias("meta_weight"))

    def equal_weighted_analysts_ls(self, book_t: pl.DataFrame) -> pl.DataFrame:
        # 1) gross par analyste
        gross_a = (
            book_t
            .group_by("analyst_id")
            .agg(pl.col("weight").abs().sum().alias("gross"))
        )

        # 2) normalise chaque analyste (gross=1)
        book_norm = (
            book_t
            .join(gross_a, on="analyst_id", how="left")
            .with_columns(
                pl.when(pl.col("gross") > 0)
                .then(pl.col("weight") / pl.col("gross"))
                .otherwise(0.0)
                .alias("w_norm")
            )
        )

        # 3) equal weight sur analystes = moyenne des books normalisés
        n = gross_a.filter(pl.col("gross") > 0).height
        ew = (
            book_norm
            .group_by("stock_id")
            .agg(pl.col("w_norm").sum().alias("meta_weight"))
        )

        if n > 0:
            ew = ew.with_columns((pl.col("meta_weight") / pl.lit(n)).alias("meta_weight"))

        # 4) renormalise le meta-book final (gross=1) pour comparabilité
        gross = ew.select(pl.col("meta_weight").abs().sum()).item()
        if gross and gross > 0:
            ew = ew.with_columns((pl.col("meta_weight") / pl.lit(gross)).alias("meta_weight"))

        return ew.sort("stock_id")

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

    def _load_pred(self):
        """
        Load the out-of-sample predictions from S3.

        Returns
        -------
        pl.DataFrame
            DataFrame containing out-of-sample predictions with columns ['date', 'model', 'analyst_id', 'y_hat'].
        """
        obj = s3Utils.pull_file_from_s3(path="s3://alpha-in-analysts-storage/results/OOS_PRED.pkl", file_type="pickle")

        df_y_hat = pd.concat(
            {model: pd.concat(d_by_date, axis=1).T
             for model, d_by_date in obj.items()},
            names=["model", "date"]
        )
        df_long = (
            df_y_hat.stack()
            .rename("y_hat")
            .reset_index()
            .rename(columns={"level_2": "id"})
        )
        return pl.DataFrame(df_long).rename({"y_hat": "predicted_pnl"})

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
    
    def _analyst_backtest(self, ptf_weights: list[pl.DataFrame], prices: pl.DataFrame, timeline: list[date], model: str) -> pl.DataFrame:
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
        model : str
            Name of the model used for the backtest.

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

        gross_return = (perf[-1, "cumulative_ret"] - 1) * 100
        annualized_return = (perf[-1, "cumulative_ret"] ** (12 / len(timeline)) - 1) * 100
        logger.info("---------------------------- Backtest Performance Summary ------------------------")
        logger.info("Model: {}".format(model))
        logger.info("Ptf Method: {}".format(self.config.construction_method))
        logger.info("Date Range: {} to {}".format(timeline[0], timeline[-1]))
        logger.info("Gross Return: {:.2f}%".format(gross_return))
        logger.info("Annualized Return: {:.2f}%".format(annualized_return))
        logger.info("----------------------------------------------------------------------------------")

        return perf