import polars as pl
import pandas as pd

# path: str = r"C:\Users\pcharbonnier\OneDrive - DVHE\Etudes\ENSAE 2025-2026\Cours\Statistical Learning\estimates.csv"
# df_est = pd.read_csv(path)
# df_est = df_est[["anndats", "amaskcd", "cusip", "cname", "value", "curr", "alysnam"]]
# df_est = df_est.rename(columns={
#     "anndats": "reco_date",
#     "amaskcd": "analyst_id",
#     "cname": "company_name",
#     "value": "target_price",
#     "curr": "currency",
#     "alysnam": "analyst_name"
# })
# df_est["reco_date"] = pd.to_datetime(df_est["reco_date"]).dt.date
# df_est["target_price"] = pd.to_numeric(df_est["target_price"], errors="coerce")
# df_est["stock_id"] = df_est["cusip"].str.slice(0, 8).str.upper()
# df_est["analyst_id"] = pd.to_numeric(df_est["analyst_id"], errors="coerce").astype("Int64").astype(str)
# df_est["analyst_name"] = df_est["analyst_name"].str.replace(r"\s+", " ", regex=True).str.strip().str.title()
# df_est["company_name"] = df_est["company_name"].str.title()
# df_est = df_est[df_est["currency"] == "USD"]
# df_est = df_est.drop(columns=["cusip", "currency"])


# df = pl.read_parquet("data/prices.parquet")

# df = df[["date", "prc", "ncusip"]]
# df = df.with_columns([
#     pl.col("ncusip").str.slice(0, 8).str.to_uppercase().alias("stock_id"),
#     pl.col("prc").abs().alias("price"),
#     pl.col("date").cast(pl.Date)
# ])
# df = df.drop(["ncusip", "prc"])

# # Filtre sur les stock présents dans le dataset des estimates pour alléger
# valid_ids = (
#     df_est["stock_id"]
#     .dropna()
#     .astype(str)
#     .unique()
#     .tolist()
# )
# df = df.filter(pl.col("stock_id").is_in(valid_ids))
# df_est = df_est.drop_nulls(subset=["stock_id", "reco_date", "target_price", "analyst_id"])

# df_est.to_parquet("data/estimates.parquet")
# df.write_parquet("data/prices.parquet")

# print(df.head(10))

df_prices = pl.read_parquet("data/prices.parquet")
df_estimates = pl.read_parquet("data/estimates.parquet")

print("test")