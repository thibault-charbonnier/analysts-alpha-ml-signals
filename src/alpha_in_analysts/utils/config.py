from dataclasses import dataclass
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """
    Configuration object to hold settings for the application.
    """
    def __init__(self):
        # PATHS
        try:
            self.ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
        except NameError:
            self.ROOT_DIR = Path.cwd()
        logger.info("Root dir: " + str(self.ROOT_DIR))
        self.RUN_CONFIG_PATH = self.ROOT_DIR / "configs"/ "run_config.json"
        logger.info("Run config path: " + str(self.RUN_CONFIG_PATH))
        self.MODEL_CONFIG_PATH = self.ROOT_DIR / "configs"/ "ml_config.json"
        logger.info("Model config path: " + str(self.MODEL_CONFIG_PATH))

        # Data path configurations
        self.target_price_path: str|None = None
        self.prices_path: str|None = None

        # Book related configurations
        self.decay_halflife: int|None = None
        self.validity_length: int|None = None

        # Portfolio construction related configurations
        self.transfo_dead_zone: float = 0.02
        self.transfo_k_pos: float = 12.0
        self.transfo_k_neg: float = 12.0
        self.construction_method: str = "paper"
        self.normalize_weights: bool = False
        self.pnl_transfo: bool = False

        # Model related configurations
        self.model_name: str|None = None
        self.model_paths: dict = {}
        self.model_params: dict = {}

        # Features related configurations
        self.start_date_features: str|None = "2002-01-31"
        self.lookback_y:int = 12
        self.lookback_perf: int = 12
        self.lookback_recent_perf: int = 6
        self.lookback_vol: int = 12
        self.lookback_recent_vol: int = 6
        self.lookback_mean_ret: int = 12
        self.lookback_recent_mean_ret: int = 6
        self.lookback_sharpe: int = 12
        self.lookback_recent_sharpe: int = 6
        self.lookback_sortino: int = 12
        self.lookback_recent_sortino: int = 6
        self.lookback_coverage: int = 12
        self.lookback_recent_coverage: int = 6

        # Backtest / Orchestrator related configurations
        self.train_start_date: str|None = None
        self.train_end_date: str|None = None
        self.warmup_train_months: int = 24

        self.test_start_date: str|None = None
        self.test_end_date: str|None = None
        self.warmup_test_months: int = 24

        # Output paths
        self.backtest_base_path: str = "None"

        self._load_run_config(filepath=self.RUN_CONFIG_PATH)
        self._load_model_config(filepath=self.MODEL_CONFIG_PATH)

    def _load_run_config(self, filepath: str|Path = "configs/run_config.json"):
        """
        Load run configuration from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the JSON configuration file.
        """
        with open(filepath, 'r') as f:
            config: dict = json.load(f)
            self.model_name = config.get('MODEL')
            self.decay_halflife = config.get('BOOK').get('DECAY_HALFLIFE_MONTHS')
            self.validity_length = config.get('BOOK').get('VALIDITY_LENGTH_MONTHS')

            self.transfo_dead_zone = config.get('PORTFOLIO').get('TRANSFO_DEAD_ZONE', 0.02)
            self.transfo_k_pos = config.get('PORTFOLIO').get('TRANSFO_K_POS', 12.0)
            self.transfo_k_neg = config.get('PORTFOLIO').get('TRANSFO_K_NEG', 12.0)
            self.construction_method = config.get('PORTFOLIO').get('METHOD', "paper")
            self.normalize_weights = config.get('PORTFOLIO').get('NORMALIZE_WEIGHTS', False)
            self.use_transfo = config.get('PORTFOLIO').get('USE_PREDICTION_TRANSFORMATION', False)

            self.start_date_features = config.get('FEATURES').get('START_DATE_FEATURES', "2002-01-31")
            self.lookback_perf = config.get('FEATURES').get('LOOKBACK_PERF', 12)
            self.lookback_recent_perf = config.get('FEATURES').get('LOOKBACK_RECENT_PERF', 6)
            self.lookback_vol = config.get('FEATURES').get('LOOKBACK_VOL', 12)
            self.lookback_recent_vol = config.get('FEATURES').get('LOOKBACK_RECENT_VOL', 6)
            self.lookback_mean_ret = config.get('FEATURES').get('LOOKBACK_MEAN_RET', 12)
            self.lookback_recent_mean_ret = config.get('FEATURES').get('LOOKBACK_RECENT_MEAN_RET', 6)
            self.lookback_sharpe = config.get('FEATURES').get('LOOKBACK_SHARPE', 12)
            self.lookback_recent_sharpe = config.get('FEATURES').get('LOOKBACK_RECENT_SHARPE', 6)
            self.lookback_sortino = config.get('FEATURES').get('LOOKBACK_SORTINO', 12)
            self.lookback_recent_sortino = config.get('FEATURES').get('LOOKBACK_RECENT_SORTINO', 6)
            self.lookback_coverage = config.get('FEATURES').get('LOOKBACK_COVERAGE', 12)
            self.lookback_recent_coverage = config.get('FEATURES').get('LOOKBACK_RECENT_COVERAGE', 6)

            self.train_start_date = config.get('RUN').get('TRAINING').get('START_DATE')
            self.train_end_date = config.get('RUN').get('TRAINING').get('END_DATE')
            self.warmup_train_months = config.get('RUN').get('TRAINING').get('WARMUP_MONTHS', 24)

            self.test_start_date = config.get('RUN').get('TESTING').get('START_DATE')
            self.test_end_date = config.get('RUN').get('TESTING').get('END_DATE')
            self.warmup_test_months = config.get('RUN').get('TESTING').get('WARMUP_MONTHS', 24)

            self.target_price_path = config.get('DATA').get('ESTIMATES_PATH')
            self.prices_path = config.get('DATA').get('PRICES_PATH')

            self.backtest_base_path = config.get('OUTPUT').get('BASE_BACKTEST', "None")

    def _load_model_config(self, filepath: str|Path = "configs/ml_config.json"):
        """
        Load model configuration from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the JSON configuration file.
        """
        with open(filepath, 'r') as f:
            config: dict = json.load(f)
            self.model_paths = config.get('PATH').get(self.model_name, {})
            self.model_params = config.get('HYPERPARAMETERS').get(self.model_name, {})