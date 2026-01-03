import logging
import polars as pl

from .book_engine import BookEngine
from .features_engine import FeaturesEngine
from .model_wrappers.model_wrapper import ModelWrapper
from .meta_portfolio import MetaPortfolio
from .utils.config import Config
from .utils.mapping import ModelType

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    High level orchestrator dedicated to manage the learning process of machine learning models on analysts' data.

    It uses high level components to perform the following tasks:
        - BookEngine: Process the analysts' target prices to generate an implied portfolio.
        - FeatureEngine: Process analysts' books to generate features for the models.
        - ModelWrapper(s): Train machine learning models on the generated features and save them for future inference.

    The traning is done as follows:
        0. Data is available from 2000 to 2025 - split into train (2000-2020) and test (2020-2025).
        1. A warm-up period is defined (e.g., 2000-2002) to allow feature generation prior to training.
        2. For each month of the real train period (2002-2020):
            a. Generate books from available analysts' target prices up to that month (x months of validity).
            b. Generate features from the generated books (current month and historical data).
            c. Stack features over time to create a training dataset.
            d. Align the target variable (e.g., 12 month future returns) with the features.
        3. Train the model one-shot on the training dataset.
        4. After training, the model is saved on disk for future inference that will be done in a separate component (Backtester).
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

    def run(self):
        """
        Run the learning process as described in the class docstring.
        """
        pass

    # -----------------------------------------------------------------
    # |                       Private Helpers                         |
    # -----------------------------------------------------------------

    def _load_model(self) -> ModelWrapper:
        """
        Instantiate the model wrapper based on the configuration.

        Returns
        -------
        ModelWrapper
            An instance of the wrapped model inherited from ModelWrapper.
        """
        self.model = ModelType[self.config.model_name].value
        self.model.create_model(hyper_params=self.config.model_params)
        logger.info(f"Model {self.config.model_name} created successfully.")
    
    def _save_model(self):
        """
        Save the trained model to disk using the ModelWrapper interface.
        """
        self.model.save_model(model_path=self.config.model_paths.get('SKOPS'),
                              manifest_path=self.config.model_paths.get('MANIFEST'))
        logger.info(f"Model {self.config.model_name} saved successfully to disk.")