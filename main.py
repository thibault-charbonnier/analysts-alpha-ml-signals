print(">>> main.py started", flush=True)
from rich.logging import RichHandler
import pandas as pd
import dataframe_image as dfi
import polars as pl
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
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
from src.alpha_in_analysts.meta_backtest import Backtester
from scripts.perf import compute_backtest_metrics_from_cumulative, plot_cumulative_perf, rolling_returns_boxplot

config = Config()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    force=True,
)
logger = logging.getLogger(__name__)
logger.info("--- Start ---")

#-------------------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------------------
train_or_load = "load"
#-------------------------------------------------------------------------------------------

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
            # Linear
            "ols": lambda **kw: LinearRegression(**kw),
            "ridge": lambda **kw: Ridge(**kw),
            "lasso": lambda **kw: Lasso(**kw),
            "elastic_net": lambda **kw: ElasticNet(**kw),
            "huber": lambda **kw: HuberRegressor(**kw),

            # Tree / Boosting
            "random_forest": lambda **kw: RandomForestRegressor(n_jobs=1, **kw),
            "xgboost": lambda **kw: XGBRegressor(
                n_jobs=1,
                objective="reg:squarederror",
                verbosity=0,
                **kw
            ),
            "lightgbm": lambda **kw: LGBMRegressor(
                n_jobs=1,
                objective="regression",
                verbosity=-1,
                **kw
            ),
            "hist_gb": lambda **kw: HistGradientBoostingRegressor(**kw),

            # NN
            "mlp": lambda **kw: MLPRegressor(**kw)
        }

    # -------------------------
    # Default hyperparameters
    # -------------------------
    if hyperparams_all_combinations is None:

        hyperparams_all_combinations = {}

        # =========================
        # OLS
        # =========================
        hyperparams_all_combinations["ols"] = [{}]

        # =========================
        # Ridge
        # =========================
        hyperparams_all_combinations["ridge"] = []
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            hyperparams_all_combinations["ridge"].append(
                {"alpha": alpha}
            )

        # =========================
        # Lasso
        # =========================
        hyperparams_all_combinations["lasso"] = []
        for alpha in [0.001, 0.01, 0.1, 1.0]:
            hyperparams_all_combinations["lasso"].append(
                {"alpha": alpha}
            )

        # =========================
        # Elastic Net
        # =========================
        hyperparams_all_combinations["elastic_net"] = []
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            for l1_ratio in [0.1, 0.5, 0.9]:
                hyperparams_all_combinations["elastic_net"].append(
                    {"alpha": alpha, "l1_ratio": l1_ratio}
                )


        # =========================
        # Huber Regressor
        # =========================
        hyperparams_all_combinations["huber"] = []
        for epsilon in [1.1, 1.35, 1.5]:
            for alpha in [0.0001, 0.001, 0.01]:
                hyperparams_all_combinations["huber"].append(
                    {
                        "epsilon": epsilon,
                        "alpha": alpha,
                        "max_iter": 10_000,
                    }
                )

        # =========================
        # Random Forest
        # =========================
        hyperparams_all_combinations["random_forest"] = []
        for n_estimators in [100, 200, 300]:
            for max_depth in [None, 10, 20]:
                for min_samples_split in [2, 5]:
                    hyperparams_all_combinations["random_forest"].append(
                        {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                        }
                    )

        # =========================
        # XGBoost
        # =========================
        hyperparams_all_combinations["xgboost"] = []
        for n_estimators in [100, 200, 300]:
            for learning_rate in [0.05, 0.1]:
                for max_depth in [3, 5]:
                    for subsample in [0.8, 1.0]:
                        hyperparams_all_combinations["xgboost"].append(
                            {
                                "n_estimators": n_estimators,
                                "learning_rate": learning_rate,
                                "max_depth": max_depth,
                                "subsample": subsample,
                            }
                        )

        # =========================
        # LightGBM
        # =========================
        hyperparams_all_combinations["lightgbm"] = []
        for num_leaves in [15, 31]:
            for learning_rate in [0.01, 0.05, 0.1]:
                for max_depth in [-1, 5, 10]:
                    hyperparams_all_combinations["lightgbm"].append(
                        {
                            "num_leaves": num_leaves,
                            "learning_rate": learning_rate,
                            "max_depth": max_depth,
                            "n_estimators": 300,
                            "min_child_samples": 20,
                            "subsample": 0.8,
                            "colsample_bytree": 0.8,
                            "random_state": 42,
                        }
                    )

        # =========================
        # Neural Network (MLP)
        # =========================
        hyperparams_all_combinations["mlp"] = []
        for hidden_layer_sizes in [(50,), (50, 50), (32,16,8)]:
            for activation in ["relu", "tanh"]:
                for alpha in [0.0001, 0.001]:
                    hyperparams_all_combinations["mlp"].append(
                        {
                            "hidden_layer_sizes": hidden_layer_sizes,
                            "activation": activation,
                            "alpha": alpha,
                        }
                    )

    # -------------------------
    # Default S3 output paths
    # -------------------------
    if path_objs_to_save is None:
        path_objs_to_save = {
            "best_score_all_models_overtime": "s3://alpha-in-analysts-storage/results/best_score_all_models_overtime_tmp.parquet",
            "best_params_all_models_overtime": "s3://alpha-in-analysts-storage/results/best_params_all_models_overtime_tmp.pkl",
            "best_hyperparams_all_models_overtime": "s3://alpha-in-analysts-storage/results/best_hyperparams_all_models_overtime_tmp.pkl",
            "OOS_PRED": "s3://alpha-in-analysts-storage/results/OOS_PRED_tmp.pkl",
            "OOS_TRUE": "s3://alpha-in-analysts-storage/results/OOS_TRUE_tmp.pkl"
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
    logger.info(f"Data loaded from S3 in {round(time.time() - start, 2)} seconds")

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
    # STORE BETAS OVER TIME (linear only)
    # -----------------------------
    linear_models = {"ridge", "lasso", "elastic_net", "ols"}

    best_params_all_models_overtime = {
        m: (
            pd.DataFrame(
                index=date_range,
                columns=["intercept"]
                        + list(
                    data.drop(columns=["date", "analyst_id", "y"]).columns
                ),
                dtype=float,
            )
            if m in linear_models
            else None
        )
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

            # Storage
            OOS_PRED[model_name][test_date] = pd.Series(
                y_hat,
                index=data[data["date"] == test_date]["analyst_id"]
            )
            OOS_TRUE[test_date] = data[data["date"] == y_true_date].set_index(
                "analyst_id"
            )["y"]

            # -----------------------------
            # STORE COEFFICIENTS (linear models only)
            # -----------------------------
            if model_name in linear_models:
                best_params_all_models_overtime[model_name].loc[date_t, "intercept"] = (
                    model_final.intercept_
                    if hasattr(model_final, "intercept_")
                    else np.nan
                )

                best_params_all_models_overtime[model_name].loc[
                    date_t,
                    X_train.columns
                ] = model_final.coef_

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
            path_objs_to_save["best_params_all_models_overtime"]: best_params_all_models_overtime,
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

def load_ml_results_from_s3():
    path_objs_to_load = {
        "best_score_all_models_overtime": "s3://alpha-in-analysts-storage/results/best_score_all_models_overtime.parquet",
        "best_params_all_models_overtime": "s3://alpha-in-analysts-storage/results/best_params_all_models_overtime.pkl",
        "best_hyperparams_all_models_overtime": "s3://alpha-in-analysts-storage/results/best_hyperparams_all_models_overtime.pkl",
        "OOS_PRED": "s3://alpha-in-analysts-storage/results/OOS_PRED.pkl",
        "OOS_TRUE": "s3://alpha-in-analysts-storage/results/OOS_TRUE.pkl",
        "features": "s3://alpha-in-analysts-storage/data/all_features.parquet"
    }
    objs_loaded = {}
    for k,v in path_objs_to_load.items():
        ext = v.split("/")[-1].split(".")[-1]
        objs_loaded[k] = s3Utils.pull_file_from_s3(
            path=v,
            file_type=ext
        )
    return objs_loaded

def get_analytics(objs_loaded:dict):
    # Plot features DF
    dfi.export(objs_loaded["features"], config.ROOT_DIR/"outputs"/"figures"/"features_df.png",max_rows=10)
    dfi.export(objs_loaded["features"], config.ROOT_DIR / "outputs" / "figures" / "features_df_short.png", max_rows=10, max_cols=10)

    # Plot best score all models overtime (validation)
    plt.figure(figsize=(10, 6))
    plt.plot(objs_loaded["best_score_all_models_overtime"])
    plt.title("Best score overtime per model")
    plt.ylabel("Score (rmse)")
    plt.ylim(top=2)
    plt.ylim(bottom=0)
    plt.xlabel("Date")
    plt.legend(objs_loaded["best_score_all_models_overtime"].columns)
    plt.grid(visible=True)
    plt.savefig(config.ROOT_DIR / "outputs" / "figures" / "best_score_all_models_overtime.png")
    plt.close()

    # Hyperparams all models overtime
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    mdl_names = list(objs_loaded["best_hyperparams_all_models_overtime"].keys())
    for k, ax in enumerate(axes.flat):
        if mdl_names[k] in ["mlp","ols"]:
            continue
        mdl = mdl_names[k]
        df = objs_loaded["best_hyperparams_all_models_overtime"][mdl]
        ax.plot(df, label=df.columns)

        ax.set_title(f"Optimal hyperparams over timer for model: {mdl}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Hyperparameter(s)")
        ax.grid(True)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(config.ROOT_DIR / "outputs" / "figures" / "best_hyperparams_all_models_overtime.png", dpi=300)
    plt.close()

    # Betas overtime – linear models
    models_params = objs_loaded["best_params_all_models_overtime"]

    n_models = len(models_params)
    n_rows, n_cols = 2, 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12), sharex=True)
    axes = axes.flatten()

    for ax, (model_name, df_params) in zip(axes, models_params.items()):

        # -----------------------------
        # Plot all coefficients
        # -----------------------------
        ax.plot(df_params)

        # -----------------------------
        # Model-specific robust y-limits
        # -----------------------------
        vals = df_params.values.flatten()
        vals = vals[~np.isnan(vals)]

        if len(vals) > 0:
            q_low, q_high = np.percentile(vals, [1, 99])
            iqr = q_high - q_low
            y_min = q_low - 0.1 * iqr
            y_max = q_high + 0.1 * iqr
            ax.set_ylim(y_min, y_max)

        # -----------------------------
        # Formatting
        # -----------------------------
        ax.set_title(f"{model_name}")
        ax.grid(True)
        ax.legend(
            df_params.columns,
            fontsize=8,
            ncol=2,
            frameon=False
        )

    # Hide unused subplots if < 4 models
    for i in range(n_models, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Optimal parameters overtime (linear models)", fontsize=16)
    fig.supxlabel("Date")
    fig.supylabel("Coefficient value")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        config.ROOT_DIR / "outputs" / "figures" / "best_parameters_all_models_overtime.png"
    )
    plt.close()

    # Proportion (overtime) of selected features linear models
    features = objs_loaded["best_params_all_models_overtime"][next(iter(objs_loaded["best_params_all_models_overtime"]))].columns
    proportion_selected_features = pd.DataFrame(
        index=features,
        columns=objs_loaded["best_params_all_models_overtime"].keys(),
        data=np.nan
    )
    for k, v in objs_loaded["best_params_all_models_overtime"].items():
        df = v.dropna(how="all")
        # is_present = round(100 * (df[df != 0.0].mean(axis=0)), 2)
        is_present = (abs(df) > 0.01).sum(axis=0)/df.shape[0]
        proportion_selected_features.loc[:, k] = round(100*is_present,2)

    proportion_selected_features["mean_models"] = round(proportion_selected_features.mean(axis=1), 2)
    proportion_selected_features = proportion_selected_features.sort_values(by="mean_models", ascending=False)
    proportion_selected_features.insert(loc=0, column="rank", value=np.arange(1, len(proportion_selected_features) + 1))
    dfi.export(proportion_selected_features,
               config.ROOT_DIR / "outputs" / "figures" / "proportion_selected_features.png")

    # Mean (time series) betas
    mean_parameters = pd.DataFrame(
        index=features,
        columns=objs_loaded["best_params_all_models_overtime"].keys(),
        data=np.nan
    )
    for k, v in objs_loaded["best_params_all_models_overtime"].items():
        mean_parameters.loc[:, k] = round(v.mean(axis=0), 2)
    mean_parameters["mean_models"] = round(mean_parameters.mean(axis=1), 2)
    mean_parameters = mean_parameters.sort_values(by="mean_models", ascending=False)
    mean_parameters.insert(loc=0, column="rank", value=np.arange(1, len(mean_parameters) + 1))
    dfi.export(mean_parameters, config.ROOT_DIR / "outputs" / "figures" / "mean_parameters.png")

    # IC plots and DF
    dates = sorted(objs_loaded["OOS_TRUE"].keys())
    models = list(objs_loaded["OOS_PRED"].keys())

    IC_df = pd.DataFrame(index=dates, columns=models, dtype=float)

    for model, preds_by_date in objs_loaded["OOS_PRED"].items():
        for date, y_pred in preds_by_date.items():

            if date not in objs_loaded["OOS_TRUE"]:
                continue

            y_true = objs_loaded["OOS_TRUE"][date]

            # Align safely on analyst_id
            df = pd.concat([y_true, y_pred], axis=1, join="inner")
            df.columns = ["y_true", "y_pred"]

            if len(df) > 1:
                ic = df["y_true"].corr(df["y_pred"], method="spearman")
                IC_df.loc[date, model] = ic

    rolling_ic = IC_df.rolling(12, min_periods=1).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_ic)
    plt.title("Rolling 12m IC overtime per model")
    plt.ylabel("IC")
    plt.ylim(top=0.25)
    plt.ylim(bottom=-0.15)
    plt.xlabel("Date")
    plt.legend(rolling_ic.columns)
    plt.grid(visible=True)
    plt.savefig(config.ROOT_DIR / "outputs" / "figures" / "ic_all_models_overtime.png")
    plt.close()

    mean_ic = pd.DataFrame(round(100 * IC_df.mean(axis=0), 2), columns=["mean_ic"])
    mean_ic = mean_ic.sort_values(by="mean_ic", ascending=False)
    mean_ic.insert(loc=0, column="rank", value=np.arange(1, len(mean_ic) + 1))
    dfi.export(mean_ic, config.ROOT_DIR / "outputs" / "figures" / "mean_ic_all_models.png")

    # OOS RMSE
    dates = sorted(objs_loaded["OOS_TRUE"].keys())
    models = list(objs_loaded["OOS_PRED"].keys())
    RMSE_df = pd.DataFrame(index=dates, columns=models, dtype=float)
    for model, preds_by_date in objs_loaded["OOS_PRED"].items():
        for date, y_pred in preds_by_date.items():

            if date not in objs_loaded["OOS_TRUE"]:
                continue

            y_true = objs_loaded["OOS_TRUE"][date]

            # Align on analyst_id
            df = pd.concat([y_true, y_pred], axis=1, join="inner")
            df.columns = ["y_true", "y_pred"]

            if len(df) > 0:
                rmse = np.sqrt(np.mean((df["y_true"] - df["y_pred"]) ** 2))
                RMSE_df.loc[date, model] = rmse

    plt.figure(figsize=(10, 6))
    plt.plot(RMSE_df)
    plt.title("OOS RMSE overtime per model")
    plt.ylabel("OOS RMSE")
    plt.ylim(top=2)
    plt.ylim(bottom=0)
    plt.xlabel("Date")
    plt.legend(RMSE_df.columns)
    plt.grid(visible=True)
    plt.savefig(config.ROOT_DIR / "outputs" / "figures" / "oos_rmse_all_models_overtime.png")
    plt.close()

    mean_oos_rmse = pd.DataFrame(round(RMSE_df.mean(axis=0), 2), columns=["mean_ic"])
    mean_oos_rmse = mean_oos_rmse.sort_values(by="mean_ic", ascending=True)
    mean_oos_rmse = mean_oos_rmse.rename(columns={"mean_ic":"mean_rmse"})
    mean_oos_rmse.insert(loc=0, column="rank", value=np.arange(1, len(mean_oos_rmse) + 1))
    dfi.export(mean_oos_rmse, config.ROOT_DIR / "outputs" / "figures" / "oos_rmse_all_models.png")

if __name__=="__main__":
    if train_or_load=="train":
        main(
            path_df_tp = "s3://alpha-in-analysts-storage/data/estimates.parquet",
            path_df_prices = "s3://alpha-in-analysts-storage/data/prices.parquet",
            path_all_features = "s3://alpha-in-analysts-storage/data/all_features.parquet",
            min_nb_periods_required = 24,
            validation_window = 12,
            forecast_horizon = 1,
            models = None,
            hyperparams_all_combinations = None,
            save_to_s3 = True,
            path_objs_to_save = None
        )
        objs_loaded = load_ml_results_from_s3()
        get_analytics(objs_loaded=objs_loaded)

        bt = Backtester(config=config)
        bt.run()

        logger.info("--- End ---")
    elif train_or_load=="load":
        objs_loaded = load_ml_results_from_s3()
        get_analytics(objs_loaded=objs_loaded)

        all_models = ["RANDOM_FOREST", "RIDGE", "LASSO", "XGBOOST", "LIGHTGBM", "OLS", "BENCHMARK", "EQUAL_WEIGHTED"]

        cfg = Config()
        bt = Backtester(config=cfg)
        df_res = bt.run(models=all_models)

        rolling_returns_boxplot(df_res, window_years=3, periods_per_year=12, annualize=True, show_fliers=False)
        plot_cumulative_perf(df_res, logy=False, title="Cumulative Performance")
        compute_backtest_metrics_from_cumulative(df_res, periods_per_year=12, risk_free_rate=0.00, as_percent=True,
                                                 round_digits=2)

        logger.info("--- End ---")
    else:
        raise ValueError("Wrong valued entered for train_or_load")

