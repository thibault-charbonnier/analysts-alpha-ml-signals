from src.alpha_in_analysts.utils.s3_utils import s3Utils

wrds_gq_monthly = s3Utils.pull_parquet_file_from_s3(
    path="s3://alpha-in-analysts-storage/data/wrds_gross_query_monthly.parquet"
)
ibes_crsp_linking_table = s3Utils.pull_parquet_file_from_s3(
    path="s3://alpha-in-analysts-storage/data/ibes_crsp_linking_table.parquet"
)
target_prices_gross_query = s3Utils.pull_parquet_file_from_s3(
    path="s3://alpha-in-analysts-storage/data/target_prices_gross_query.parquet"
)

# Ex to push
# s3Utils.push_object_to_s3_parquet(
#     object_to_push=dataframe_to_push,
#     path="s3://alpha-in-analysts-storage/data/wrds_gross_query_monthly.parquet"
# )
