import polars as pl
import numpy as np

class Backtest:
    """Class to backtest a strategy"""
    def __init__(self,
                 returns:pl.DataFrame,
                 weights:pl.DataFrame,
                 turnover:pl.DataFrame,
                 transaction_costs:int|float=10,
                 strategy_name:str=""):
        self.returns = returns
        self.weights = weights
        self.portfolio_gross_returns = None
        self.portfolio_net_returns = None
        self.cropped_portfolio_gross_returns = None
        self.cropped_portfolio_net_returns = None
        self.transaction_costs = transaction_costs
        self.turnover = turnover
        self.start_date = None
        self.strategy_name = strategy_name

    def run_backtest(self) -> None:
        """Run the backtest (Polars version)"""

        weights_cols = pl.all().exclude("date")

        # --------------------------------------------------
        # Gross portfolio returns: (returns * weights).sum(axis=1)
        # --------------------------------------------------
        asset_cols = [c for c in self.returns.columns if c != "date"]

        df = (
            self.returns
            .join(self.weights, on="date", how="inner", suffix="_w")
        )

        portfolio_expr = (
            pl.when(
                # Case 1: all weights are NaN → portfolio = NaN
                pl.all_horizontal([pl.col(f"{c}_w").is_null() for c in asset_cols])
            )
            .then(pl.lit(None))
            .otherwise(
                pl.when(
                    # Case 2: no valid (w, r) pair → NaN
                    pl.all_horizontal([
                        pl.col(c).is_null() | pl.col(f"{c}_w").is_null()
                        for c in asset_cols
                    ])
                )
                .then(pl.lit(None))
                .otherwise(
                    # Case 3: sum valid contributions
                    pl.sum_horizontal(
                        *[
                            pl.when(
                                pl.col(c).is_not_null() & pl.col(f"{c}_w").is_not_null()
                            )
                            .then(pl.col(c) * pl.col(f"{c}_w"))
                            .otherwise(None)
                            for c in asset_cols
                        ]
                    )
                )
            )
        )

        self.portfolio_gross_returns = (
            df
            .with_columns(portfolio_expr.alias(self.strategy_name))
            .select(["date", self.strategy_name])
        )

        # --------------------------------------------------
        # Net returns initialization
        # --------------------------------------------------
        self.portfolio_net_returns = self.portfolio_gross_returns.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(self.strategy_name)
        )

        # --------------------------------------------------
        # Transaction costs
        # tc = transaction_costs * (turnover / 10000)
        # --------------------------------------------------
        tc = (
            self.turnover
            .with_columns(
                (self.transaction_costs * (pl.col("turnover") / 10000))
                .fill_null(0.0)
                .alias("tc")
            )
            .select(["date", "tc"])
        )

        # --------------------------------------------------
        # Net returns = gross - tc
        # --------------------------------------------------
        self.portfolio_net_returns = (
            self.portfolio_gross_returns
            .join(tc, on="date", how="left")
            .with_columns(
                (pl.col(self.strategy_name) - pl.col("tc"))
                .alias(self.strategy_name)
            )
            .select(["date", self.strategy_name])
        )

        # --------------------------------------------------
        # Start date: first date with at least one non-null weight
        # --------------------------------------------------
        self.start_date = (
            self.weights
            .with_columns(
                pl.any_horizontal(weights_cols).alias("has_weight")
            )
            .filter(pl.col("has_weight"))
            .select(pl.col("date").min())
            .item()
        )

        # --------------------------------------------------
        # Crop results
        # --------------------------------------------------
        self.cropped_portfolio_gross_returns = (
            self.portfolio_gross_returns
            .filter(pl.col("date") >= self.start_date)
        )

        self.cropped_portfolio_net_returns = (
            self.portfolio_net_returns
            .filter(pl.col("date") >= self.start_date)
        )

    def get_results(self)->None:
        """Get the backtest results"""
        if self.portfolio_gross_returns is None or self.portfolio_net_returns is None:
            self.run_backtest()
        return