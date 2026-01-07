import pandas as pd
import polars as pl
import time
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from dotenv import load_dotenv
load_dotenv()
import logging
import sys
from src.alpha_in_analysts.utils.s3_utils import s3Utils
from src.alpha_in_analysts.features_engine import FeaturesEngine
from src.alpha_in_analysts.utils.config import Config

def main(
    path_df_tp: str = "s3://alpha-in-analysts-storage/data/estimates.parquet",
    path_df_prices: str = "s3://alpha-in-analysts-storage/data/prices.parquet",
    path_all_features: str = "s3://alpha-in-analysts-storage/data/all_features.parquet",
    min_nb_periods_required: int = 24,
    validation_window: int = 12,
    forecast_horizon: int = 1,
    models: dict | None = None,
    hyperparams_all_combinations: dict | None = None,
    save_to_s3: bool = True,
    path_objs_to_save: dict | None = None,
):
    # -------------------------
    # Default models
    # -------------------------
    if models is None:
        models = {
            "ridge": Ridge,
            "lasso": Lasso,
            "random_forest": RandomForestRegressor(n_jobs=1)
        }

    # -------------------------
    # Default hyperparameters
    # -------------------------
    if hyperparams_all_combinations is None:
        hyperparams_all_combinations = {
            "ridge": [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}],
            "lasso": [{"alpha": 0.01}, {"alpha": 0.1}, {"alpha": 1.0}],
            "random_forest": [
                {"n_estimators": n, "max_depth": d, "random_state": 0}
                for n in [50, 100]
                for d in [5, 10]
            ]
        }
        # hyperparams_all_combinations = {}
        #
        # # =========================
        # # Ridge
        # # =========================
        # hyperparams_all_combinations["ridge"] = []
        # for alpha in [0.01, 0.1, 1.0, 10.0]:
        #     hyperparams_all_combinations["ridge"].append(
        #         {"alpha": alpha}
        #     )
        #
        # # =========================
        # # Lasso
        # # =========================
        # hyperparams_all_combinations["lasso"] = []
        # for alpha in [0.001, 0.01, 0.1, 1.0]:
        #     hyperparams_all_combinations["lasso"].append(
        #         {"alpha": alpha}
        #     )
        #
        # # =========================
        # # Elastic Net
        # # =========================
        # hyperparams_all_combinations["elastic_net"] = []
        # for alpha in [0.001, 0.01]:
        #     for l1_ratio in [0.2, 0.5, 0.8]:
        #         hyperparams_all_combinations["elastic_net"].append(
        #             {"alpha": alpha, "l1_ratio": l1_ratio}
        #         )
        #
        # # =========================
        # # Random Forest
        # # =========================
        # hyperparams_all_combinations["random_forest"] = []
        # for n_estimators in [100, 200, 300]:
        #     for max_depth in [None, 10, 20]:
        #         for min_samples_split in [2, 5]:
        #             hyperparams_all_combinations["random_forest"].append(
        #                 {
        #                     "n_estimators": n_estimators,
        #                     "max_depth": max_depth,
        #                     "min_samples_split": min_samples_split,
        #                 }
        #             )
        #
        # # =========================
        # # XGBoost
        # # =========================
        # hyperparams_all_combinations["xgboost"] = []
        # for n_estimators in [100, 200, 300]:
        #     for learning_rate in [0.05, 0.1]:
        #         for max_depth in [3, 5]:
        #             for subsample in [0.8, 1.0]:
        #                 hyperparams_all_combinations["xgboost"].append(
        #                     {
        #                         "n_estimators": n_estimators,
        #                         "learning_rate": learning_rate,
        #                         "max_depth": max_depth,
        #                         "subsample": subsample,
        #                     }
        #                 )
        #
        # # =========================
        # # Neural Network (MLP)
        # # =========================
        # hyperparams_all_combinations["mlp"] = []
        # for hidden_layer_sizes in [(50,), (100,), (50, 50), (100, 50)]:
        #     for activation in ["relu", "tanh"]:
        #         for alpha in [0.0001, 0.001]:
        #             hyperparams_all_combinations["mlp"].append(
        #                 {
        #                     "hidden_layer_sizes": hidden_layer_sizes,
        #                     "activation": activation,
        #                     "alpha": alpha,
        #                 }
        #             )

    # -------------------------
    # Default S3 output paths
    # -------------------------
    if path_objs_to_save is None:
        path_objs_to_save = {
            "best_score_all_models_overtime": "s3://alpha-in-analysts-storage/results/best_score_all_models_overtime.parquet",
            "best_hyperparams_all_models_overtime": "s3://alpha-in-analysts-storage/results/best_hyperparams_all_models_overtime.pickle",
            "OOS_PRED": "s3://alpha-in-analysts-storage/results/OOS_PRED.pickle",
            "OOS_TRUE": "s3://alpha-in-analysts-storage/results/OOS_TRUE.pickle",
        }

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Pipeline logic
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # Config Logger
    #-----------------------------------------------------------------------------
    logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    logger = logging.getLogger(__name__)

    #-----------------------------------------------------------------------------
    # Create features
    #-----------------------------------------------------------------------------
    config = Config()
    start = time.time()
    df_tp = s3Utils.pull_parquet_file_from_s3(path=path_df_tp, to_polars=True)
    df_prices = s3Utils.pull_parquet_file_from_s3(path=path_df_prices, to_polars=True)
    logger.info("Data loaded from S3 in", round(time.time() - start, 2), "seconds")

    feature_engine = FeaturesEngine(
        config=config,
        df_prices=df_prices,
        df_tp=df_tp
    )

    try:
        all_features = s3Utils.pull_parquet_file_from_s3(
            path=path_all_features
        )
    except Exception as e:
        logger.info("all_features not on s3, computing it.")
        for i, date in enumerate(feature_engine.dates[min_nb_periods_required:]):  # skip first 24 months to have enough lookback
            logger.info("Building features at date:", date)
            res = feature_engine.get_features_and_y(
                up_to_date=str(date)
            )
            if i==0:
                all_features = res
            else:
                all_features = pl.concat([all_features, res], how="vertical")
    logger.info(all_features)

    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    # ML models
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    # -----------------------------
    # DATA
    # -----------------------------
    if not isinstance(all_features, pd.DataFrame):
        data = all_features.to_pandas()
    else:
        data = all_features
    date_range = np.sort(data["date"].unique())

    # -----------------------------
    # STORAGE (panel-safe)
    # -----------------------------
    OOS_PRED = {m: {} for m in models}
    OOS_TRUE = {}

    best_score_all_models_overtime = pd.DataFrame(
        index=date_range, columns=list(models.keys())
    )

    best_hyperparams_all_models_overtime = {
        m: pd.DataFrame(index=date_range, columns=list(hyperparams_all_combinations[m][0].keys()))
        for m in models
    }

    # -----------------------------
    # WALK-FORWARD LOOP
    # -----------------------------
    start_idx = min_nb_periods_required + validation_window + forecast_horizon

    for t in range(start_idx, len(date_range) - forecast_horizon):
        date_t = date_range[t]
        logger.info(f"Training models for date {date_t} (t={t}) {t}/{len(date_range) - forecast_horizon - 1}")
        # Compute time to do one loop
        start = time.time()

        # -----------------------------
        # SPLITS
        # -----------------------------
        train_end = date_range[t - validation_window - forecast_horizon]
        val_end   = date_range[t - forecast_horizon]

        train_data = data[data["date"] <= train_end]
        val_data = data[(data["date"] > train_end) & (data["date"] <= val_end)]

        X_train = train_data.drop(columns=["date", "analyst_id", "y"])
        y_train = train_data["y"]

        # -----------------------------
        # MODEL LOOP
        # -----------------------------
        for model_name, ModelClass in models.items():
            start_model = time.time()

            best_score = -np.inf
            best_hyperparams = None

            # -----------------------------
            # HYPERPARAMETER SELECTION
            # -----------------------------
            # =========================
            # HYPERPARAMETER FUNCTION
            # =========================
            def evaluate_hyperparams(hyperparams):
                start_hyperparams = time.time()

                model = ModelClass(**hyperparams)
                model.fit(X=X_train, y=y_train)

                ICs = []
                for d in np.sort(val_data["date"].unique()):
                    X_val = val_data[val_data["date"] == d].drop(
                        columns=["date", "analyst_id", "y"]
                    )
                    y_val = val_data[val_data["date"] == d]["y"]

                    y_hat = model.predict(X_val)

                    if len(y_val) > 1:
                        ic = np.sqrt(mean_squared_error(y_val, y_hat))
                        if not np.isnan(ic):
                            ICs.append(ic)

                if len(ICs) == 0:
                    return None

                score = np.mean(ICs)

                logger.info(
                    f"Loop finished for {model_name} / {hyperparams} "
                    f"in: {round((time.time() - start_hyperparams) / 60, 4)} min"
                )

                return score, hyperparams

            # =========================
            # PARALLEL GRID SEARCH
            # =========================
            results = Parallel(n_jobs=-1)(
                delayed(evaluate_hyperparams)(hyperparams)
                for hyperparams in hyperparams_all_combinations[model_name]
            )

            best_score = -np.inf
            best_hyperparams = None

            for res in results:
                if res is None:
                    continue
                score, hyperparams = res
                if score > best_score:
                    best_score = score
                    best_hyperparams = hyperparams

            # -----------------------------
            # STORE VALIDATION RESULTS
            # -----------------------------
            best_score_all_models_overtime.loc[date_t, model_name] = best_score
            if best_hyperparams is not None:
                for k, v in best_hyperparams.items():
                    best_hyperparams_all_models_overtime[model_name].loc[date_t, k] = v

            # -----------------------------
            # FINAL TRAINING
            # -----------------------------
            full_train = data[data["date"] <= val_end]

            model_final = ModelClass(**best_hyperparams)
            model_final.fit(
                X=full_train.drop(columns=["date", "analyst_id", "y"]),
                y=full_train["y"]
            )

            test_date = date_range[t]
            y_true_date = date_range[t]

            X_test = data[data["date"] == test_date].drop(
                columns=["date", "analyst_id", "y"]
            )
            y_hat = model_final.predict(X_test)

            OOS_PRED[model_name][test_date] = pd.Series(
                y_hat,
                index=data[data["date"] == test_date]["analyst_id"]
            )
            OOS_TRUE[test_date] = data[data["date"] == y_true_date].set_index(
                "analyst_id"
            )["y"]

            logger.info(
                f"Loop finished for {model_name} in: "
                f"{round((time.time() - start_model) / 60, 4)} min"
            )

        logger.info(
            f"Loop finished in: {round((time.time() - start) / 60, 4)} min"
        )

    # Saving results to s3
    if save_to_s3:
        objs = {
            path_objs_to_save["best_score_all_models_overtime"]: best_score_all_models_overtime,
            path_objs_to_save["best_hyperparams_all_models_overtime"]: best_hyperparams_all_models_overtime,
            path_objs_to_save["OOS_PRED"]: OOS_PRED,
            path_objs_to_save["OOS_TRUE"]: OOS_TRUE,
        }
        for path, obj in objs.items():
            ext = path.split("/")[-1].split(".")[-1]
            s3Utils.push_object_to_s3(
                object_to_push=obj,
                path=path,
                file_type=ext
            )
    else:
        logger.info("Forecasts done. End of script")

if __name__=="__main__":
    main()










