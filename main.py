print(">>> main.py started", flush=True)

import polars as pl
import time
from src.alpha_in_analysts.utils.s3_utils import s3Utils
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    force=True,
)
logger = logging.getLogger(__name__)

# logger.info("Run launched")

# logger.info("Loading data from S3...")

# start = time.time()
# df_tp = s3Utils.pull_parquet_file_from_s3(path="s3://alpha-in-analysts-storage/data/estimates.parquet", to_polars=True)
# df_prices = s3Utils.pull_parquet_file_from_s3(path="s3://alpha-in-analysts-storage/data/prices.parquet", to_polars=True)
    
# logger.info("Data loaded from S3 in %s seconds", round(time.time() - start, 2))

# from src.alpha_in_analysts.book_engine import BookEngine

# start = time.time()
# engine = BookEngine(df_tp=df_tp, df_prices=df_prices)
# df_book = engine.at_snapshot(snapshot_date="2023-12-31")

# logger.info("Book computed at snapshot date in %s seconds", round(time.time() - start, 2))

from src.alpha_in_analysts.backtester import Backtester
from src.alpha_in_analysts.utils.config import Config

logger.info("--- Start ---")

cfg = Config()
bt = Backtester(config=cfg)
bt.run()

logger.info("--- End ---")