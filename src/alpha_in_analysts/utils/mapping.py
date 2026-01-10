from enum import Enum
from ..model_wrappers import *

class ModelType(Enum):
    RANDOM_FOREST = RandomForestWrapper
    ELASTIC_NET = ElasticNetWrapper

map_model = {
    "RANDOM_FOREST": "random_forest",
    "RIDGE": "ridge",
    "LASSO": "lasso",
    "NEURAL_NETWORK": "mlp",
    "XGBOOST": "xgboost",
    "LIGHTGBM": "lightgbm",
    "OLS": "ols"
}