import polars as pl
import logging

from .book_engine import BookEngine
from .features_engine import FeaturesEngine
from .model_wrappers.model_wrapper import ModelWrapper
from .meta_portfolio import MetaPortfolio
from .utils.config import Config
from .utils.mapping import ModelType

logger = logging.getLogger(__name__)


class Backtester:
    """
    High level orchestrator dedicated to the backtesting process of the strategy based on machine learning models.

    It uses high level components to perform the following tasks:
        - BookEngine: Process the analysts' target prices to generate an implied portfolio.
        - FeatureEngine: Process analysts' books to generate features for the models.
        - ModelWrapper(s): Load trained machine learning models to perform inference on the generated features. 
        - MetaPortfolio: Construct and evaluate the strategy portfolio based on model predictions with a given methodology.

    Note: The models are assumed to be already trained and saved on disk by a separate component (Orchestrator).
          This part of training and saving is not handled here and is considered abstracted via the wrapper interface.
          Whatever model is used, it should be loadable and usable via the ModelWrapper interface.

    The backtesting is done as follows:
        0. Data is available from 2000 to 2025 - split into train (2000-2020) and test (2020-2025).
        1. A warm-up period is defined (e.g., 2018-2020) to allow feature generation prior to backtesting.
        2. For each month of the test period (2020-2025):
            a. Generate books from available analysts' target prices up to that month (x months of validity).
            b. Generate features from the generated books (current month and historical data).
            c. Load the trained model from disk.
            d. Perform inference on the generated features to obtain predictions.
            e. Construct the strategy portfolio based on model predictions using a defined methodology.
        3. Analyze the performance of the strategy over the backtest period using various metrics and visualizations.
    """

    def __init__(self, config: Config):
        """
        Parameters
        ----------
        config : Config
            Configuration object holding settings for the backtesting process.
        """
        self.config = config
        self.model: ModelWrapper = None

        self._load_model()

    def run(self):
        """
        Run the backtesting process as described in the class docstring.
        """
        pass
    
    # -----------------------------------------------------------------
    # |                       Private Helpers                         |
    # -----------------------------------------------------------------

    def _load_model(self) -> ModelWrapper:
        """
        Load the trained model from disk using the ModelWrapper interface.

        Returns
        -------
        ModelWrapper
            An instance of the wrapped model loaded from disk inherited from ModelWrapper.
        """
        self.model = ModelType[self.config.model_name].value
        self.model.load_model(model_path=self.config.model_paths.get('SKOPS'),
                              manifest_path=self.config.model_paths.get('MANIFEST'))
        logger.info(f"Model {self.config.model_name} loaded successfully from disk.")