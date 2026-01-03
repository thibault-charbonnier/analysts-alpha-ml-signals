from dataclasses import dataclass
import json


@dataclass
class Config:
    """
    Configuration object to hold settings for the application.
    """
    def __init__(self):
        # Data path configurations
        self.target_price_path: str = None
        self.prices_path: str = None

        # Book related configurations
        self.decay_halflife: int = None
        self.validity_length: int = None

        # Portfolio construction related configurations
        self.transfo_dead_zone: float = 0.02
        self.transfo_k_pos: float = 12.0
        self.transfo_k_neg: float = 12.0
        self.construction_method: str = "paper",
        self.normalize_weights: bool = False

        # Model related configurations
        self.model_name: str = None
        self.model_paths: dict = {}
        self.model_params: dict = {}

        # Backtest / Orchestrator related configurations
        self.train_start_date: str = None
        self.train_end_date: str = None
        self.warmup_train_months: int = 24

        self.test_start_date: str = None
        self.test_end_date: str = None
        self.warmup_test_months: int = 24

        # Output paths
        self.backtest_base_path: str = "None"

        self._load_run_config()
        self._load_model_config()

    def _load_run_config(self, filepath: str = "configs/run_config.json"):
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

            self.train_start_date = config.get('RUN').get('TRAINING').get('START_DATE')
            self.train_end_date = config.get('RUN').get('TRAINING').get('END_DATE')
            self.warmup_train_months = config.get('RUN').get('TRAINING').get('WARMUP_MONTHS', 24)

            self.test_start_date = config.get('RUN').get('TESTING').get('START_DATE')
            self.test_end_date = config.get('RUN').get('TESTING').get('END_DATE')
            self.warmup_test_months = config.get('RUN').get('TESTING').get('WARMUP_MONTHS', 24)

            self.target_price_path = config.get('DATA').get('ESTIMATES_PATH')
            self.prices_path = config.get('DATA').get('PRICES_PATH')

            self.backtest_base_path = config.get('OUTPUT').get('BASE_BACKTEST', "None")

    def _load_model_config(self, filepath: str = "configs/ml_config.json"):
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