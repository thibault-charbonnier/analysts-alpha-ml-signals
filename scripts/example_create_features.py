import polars as pl
import time
from dotenv import load_dotenv
from src.alpha_in_analysts.utils.s3_utils import s3Utils

load_dotenv()
start = time.time()
df_tp = s3Utils.pull_parquet_file_from_s3(path="s3://alpha-in-analysts-storage/data/estimates.parquet", to_polars=True)
df_prices = s3Utils.pull_parquet_file_from_s3(path="s3://alpha-in-analysts-storage/data/prices.parquet", to_polars=True)

print("Data loaded from S3 in", round(time.time() - start, 2), "seconds")

from src.alpha_in_analysts.book_engine import BookEngine
from src.alpha_in_analysts.features_engine import FeaturesEngine

start = time.time()

feature_engine = FeaturesEngine(
    df_prices=df_prices,
    df_tp=df_tp,
    validity_length=12,
    decay_half_life=6,
    start_date="2000-01-31",
    end_date="2024-12-31",
)
# feature_engine._build_y()
# feature_engine._build_pnl_all_analysts()
# feature_engine.pnl_all_analysts.write_parquet("pnl_all_analysts_aligned_ret_ffill2m.parquet")
d = s3Utils.pull_parquet_file_from_s3("s3://alpha-in-analysts-storage/data/pnl_all_analysts.parquet",
                                  to_polars=True)
dates = d.select(pl.col("date").unique()).sort("date").to_series().to_list()
for i, date in enumerate(dates[24:]):  # skip first 24 months to have enough lookback
    print("Building features up to date:", date)
    res = feature_engine.get_features_and_y(up_to_date=str(date),
                                            lookback_perf_pct=12,
                                            lookback_perf=6,
                                            lookback_vol_pct=6,
                                            lookback_vol = 6,
                                            lookback_mean_ret=6,
                                            lookback_sharpe=12,
                                            lookback_recent_sharpe=6,
                                            lookback_sortino=12,
                                            lookback_recent_sortino=6,
                                            lookback_y=12
                                            )
    if i==0:
        all_features = res
    else:
        all_features = pl.concat([all_features, res], how="vertical")


