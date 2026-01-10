import polars as pl
import logging

logger = logging.getLogger(__name__)


class MetaPortfolio:
    """
    Class defining a meta-portfolio that aggregates multiple analyst books into a single portfolio.

    The idea, at a given time t:
        - Each analyst provides a portfolio of weights for a set of assets (a significantly smaller subset of the full universe).
        - Each analyst also has a predicted 12-month forward PnL for their portfolio based on a machine learning signal.
        - This signal can be transformed into a more complex score that reflects the credibility of the analyst (double sigmoid for example).
        - The meta-portfolio aggregates these analyst portfolios into a single portfolio by weighting each analyst's portfolio by their credibility score.
    """

    def __init__(self, dead_zone: float = 0.02, k_pos: float = 12.0, k_neg: float = 12.0):
        self.dead_zone = dead_zone
        self.k_pos = k_pos
        self.k_neg = k_neg

    def create_metaportfolio(
            self,
            analyst_books: pl.DataFrame,
            predict_pnl: pl.DataFrame,
            method = "paper",
            normalize: bool = False,
            use_transfo: bool = True
        ) -> pl.DataFrame:
        """
        Create a meta-portfolio with individual analyst books and their predicted PnL.

        Two methods available:
            - 'paper': as in the original paper done in analyst-level and normalize the signals on each side (positive and negative)
            - 'aggregation': asset-level aggregation of weighted analyst books (more common way)

        Parameters
        ----------
        analyst_books : pl.DataFrame
            Polars DataFrame containing the portfolios of multiple analysts.
            Expected columns: ['analyst_id', 'stock_id', 'weight']
        predict_pnl : pl.DataFrame
            Polars DataFrame containing the predicted 12-month forward PnL for each analyst.
            Expected columns: ['analyst_id', 'predicted_pnl']
        method : str, default="paper"
            Method to use for constructing the meta-portfolio. Options are 'paper' or 'aggregation
        normalize : bool, default=False
            Whether to normalize the final meta-portfolio weights so that the sum of absolute weights equals 1.0
        use_transfo: bool, default=True
            Whether to apply the credibility transformation to the predicted PnL before constructing the meta-portfolio

        Returns
        -------
        pl.DataFrame
            Polars DataFrame containing the aggregated meta-portfolio weights.
            Columns: ['stock_id', 'meta_weight']
        """
        analyst_cols = ["analyst_id", "stock_id", "weight"]
        for col in analyst_cols:
            if col not in analyst_books.columns:
                raise ValueError(f"Column '{col}' not found in analyst_books DataFrame.")
        analyst_books = analyst_books.select(analyst_cols)
        
        predict_cols = ["analyst_id", "predicted_pnl"]
        for col in predict_cols:
            if col not in predict_pnl.columns:
                raise ValueError(f"Column '{col}' not found in predict_pnl DataFrame.")
        predict_pnl = predict_pnl.select(predict_cols)

        if analyst_books.is_empty() or predict_pnl.is_empty():
            raise ValueError("Input DataFrames cannot be empty.")
        
        if method == "paper":
            df = self._build_paper(analyst_books, predict_pnl, use_transfo)
        elif method == "aggregation":
            df = self._build_aggregation(analyst_books, predict_pnl, use_transfo)
        else:
            raise ValueError("method parameter must be 'paper' or 'aggregation'")

        if normalize:
            total_abs_weight = df.select(pl.col("meta_weight").abs().sum()).item() or 1.0
            if total_abs_weight == 0.0:
                raise ValueError("Total absolute weight of the meta-portfolio is zero, cannot normalize.")
            df = df.with_columns((pl.col("meta_weight") / pl.lit(total_abs_weight)).alias("meta_weight"))
            
        return df
    
    def _build_aggregation(self, analyst_books: pl.DataFrame, predict_pnl: pl.DataFrame, use_transfo: bool) -> pl.DataFrame:
        """
        Build the meta-portfolio using asset-level aggregation where the meta weight for each asset is:
            w_raw(k) = Σ_i c_i * w_{i,k}
        Where:
            - i is the analyst index
            - k the asset index.
            - c_i is the credibility score for analyst i

        Methodology:
            1. Join analyst books with their predicted PnL.
            2. Compute credibility scores c_i
            3. Compute raw weights w_raw(k) for each asset by summing c_i * w_{i,k} across all analysts.
            4. Aggregate at the asset level.

        Parameters
        ----------
        analyst_books : pl.DataFrame
            Pre-validated with columns: ['analyst_id', 'stock_id', 'weight']
        predict_pnl : pl.DataFrame
            Pre-validated with columns: ['analyst_id', 'predicted_pnl']
        use_transfo : bool
            Whether to apply the credibility transformation to the predicted PnL before constructing the meta-portfolio

        Returns
        -------
        pl.DataFrame
            Columns: ['stock_id', 'meta_weight']
        """
        signal_expr = (
            self._credibility(pl.col("predicted_pnl"), self.dead_zone, self.k_pos, self.k_neg)
            if use_transfo
            else pl.col("predicted_pnl")
        )

        return (
            analyst_books
            .join(
                predict_pnl, on="analyst_id", how="inner"
            )
            .with_columns(
                signal_expr.alias("signal"),
            )
            .with_columns(
                (pl.col("weight") * pl.col("signal")).alias("w_raw")
            )
            .group_by("stock_id")
            .agg(pl.col("w_raw").sum().alias("meta_weight"))
            .select(["stock_id", "meta_weight"])
        )

    def _build_paper(self, analyst_books: pl.DataFrame, predict_pnl: pl.DataFrame, use_transfo: bool) -> pl.DataFrame:
        """
        Build the meta-portfolio using analyst-portfolio-level as in the original paper:

        Methodology:
            1. Compute credibility scores
            2. Split analysts into positive and negative signal groups c_i^+ and c_i^-
            3. Normalize c_i^+ and c_i^- so that Σ_i c_i^+ = 1 and Σ_i c_i^- = 1 -> Same amplitude on signal on both sides (not weights)
            4. For each analyst, compute their contribution to each asset:
                - Long side (positive signal): + c_i^+ * w_{i,k}
                - Short side (negative signal): - c_i^- * w_{i,k}
            5. Aggregate by asset to get final meta weights of the portfolio.

        Parameters
        ----------
        analyst_books : pl.DataFrame
            Pre-validated with columns: ['analyst_id', 'stock_id', 'weight']
        predict_pnl : pl.DataFrame
            Pre-validated with columns: ['analyst_id', 'predicted_pnl']
        use_transfo : bool
            Whether to apply the credibility transformation to the predicted PnL before constructing the meta-portfolio

        Returns
        -------
        pl.DataFrame
            Columns: ['stock_id', 'meta_weight']
        """
        signal_expr = (
            self._credibility(pl.col("predicted_pnl"), self.dead_zone, self.k_pos, self.k_neg)
            if use_transfo
            else pl.col("predicted_pnl")
        )

        sig = (
            predict_pnl
            .with_columns(
                signal_expr.alias("signal"),
            )
            .with_columns(
                pl.when(pl.col("signal") > 0).then(pl.col("signal")).otherwise(0.0).alias("sig_pos"),
                pl.when(pl.col("signal") < 0).then(-pl.col("signal")).otherwise(0.0).alias("sig_neg"),
            )
        )
        sum_pos = sig.select(pl.col("sig_pos").sum()).item() or 0.0
        sum_neg = sig.select(pl.col("sig_neg").sum()).item() or 0.0

        scalars = sig.with_columns([
            (pl.when(pl.lit(sum_pos) > 0).then(pl.col("sig_pos") / pl.lit(sum_pos)).otherwise(0.0)).alias("c_pos"),
            (pl.when(pl.lit(sum_neg) > 0).then(pl.col("sig_neg") / pl.lit(sum_neg)).otherwise(0.0)).alias("c_neg"),
        ]).select(["analyst_id", "c_pos", "c_neg"])


        long_part = (
            analyst_books.join(scalars.select(["analyst_id", "c_pos"]), on="analyst_id", how="inner")
                         .with_columns((pl.col("weight") * pl.col("c_pos")).alias("w_contrib"))
                         .group_by("stock_id")
                         .agg(pl.col("w_contrib").sum().alias("w_pos"))
        )

        short_part = (
            analyst_books.join(scalars.select(["analyst_id", "c_neg"]), on="analyst_id", how="inner")
                         .with_columns((-pl.col("weight") * pl.col("c_neg")).alias("w_contrib"))
                         .group_by("stock_id")
                         .agg(pl.col("w_contrib").sum().alias("w_neg"))
        )

        return (
            long_part.join(short_part, on="stock_id", how="outer")
                     .with_columns([
                         pl.col("w_pos").fill_null(0.0),
                         pl.col("w_neg").fill_null(0.0),
                     ])
                     .with_columns((pl.col("w_pos") + pl.col("w_neg")).alias("meta_weight"))
                     .select(["stock_id", "meta_weight"])
        )
    
    @staticmethod
    def _credibility(x: pl.Expr, dead_zone: float = 0.02, k_pos: float = 12.0, k_neg: float = 12.0) -> pl.Expr:
        """
        Signal transformation to compute analyst credibility scores by applying a double sigmoid function.

        This mapping enhances the differentiation between high and low credibility analysts:
            - High positive predicted PnL analysts receive a higher positive credibility score (Long).
            - High negative predicted PnL analysts receive a higher negative credibility score (Short).
            - Low negative/positive predicted PnL analysts are pushed closer to zero credibility
              (Either Long or Short weights but 0 or very low credibility).
        
        By doing this transformation, we consider only the highest signals to be truly credible in a long/short portfolio.
        One can adjust the amplitude of this transformation to be more or less selective (more or less "trusting" mid signals).

        An aggresive transformation would be:
            - dead_zone = 0.05  (5% of predicted PnL considered neutral on each side so between -5% and +5%)
            - k_pos = 15.0      (PnL are considered very credible as soon as they exceed the dead_zone)
            - k_neg = 15.0      (PnL are considered very credible as soon as they exceed the dead_zone)

        A more conservative transformation would be:
            - dead_zone = 0.01  (1% of predicted PnL considered neutral on each side so between -1% and +1%)
            - k_pos = 8.0       (PnL need to be significantly above the dead_zone to be considered credible)
            - k_neg = 8.0       (PnL need to be significantly below the dead_zone to be considered credible)

        Parameters
        ----------
        x : pl.Expr
            Polars columns containing the predicted PnL values for analysts.
        dead_zone : float, default=0.02
            Half-width of the neutral zone around 0 where credibility is set to 0.
        k_pos : float, default=12.0
            Steepness of the positive-side sigmoid (x > +dead_zone).
            Higher value means sharper transition and larger scores for the same x.
        k_neg : float, default=12.0
            Steepness of the negative-side sigmoid (x < -dead_zone).
            Higher value means sharper transition and more negative scores for the same x.

        Returns
        -------
        pl.Expr
            Polars columns containing the transformed credibility scores.
        """
        pos = 1 / (1 + (-k_pos * (x - dead_zone)).exp()) - 0.5
        neg = -(1 / (1 + (-k_neg * (-x - dead_zone)).exp()) - 0.5)

        return (
            pl.when(x.is_null()).then(0.0)
            .when(x > dead_zone).then(pos)
            .when(x < -dead_zone).then(neg)
            .otherwise(0.0)
        )
