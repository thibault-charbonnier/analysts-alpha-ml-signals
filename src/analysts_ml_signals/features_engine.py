import polars as pl


class FeaturesEngine:
    """
    High-level module responsible for building analysts's features.
    Those features are use to predict analysts's futur realised PnLs.

    Predicted futur PnLs or implied PnLs will then be use as a reliability metric
    for analysts. Analysts that have a high predicted PnLs are consitered to be a 
    source of information for which we want to be long. Their implied portfolios
    will have a positive weight in the global metaportfolio of analysts.
    In the contrary, analysts with a negative predicted PnLs will be shorted.

    Engineered features are of different kinds:
        - Convictions
        - Reco age and revisions
        - Coverage
        - Turnover
        - Track-record
        - Comparison / ranking
    """

    def __init__(self, df_book: pl.DataFrame):
        ...

    def build_features(self):
        ...

    def _from_trackrecord(self):
        ...

    def _from_book(self):
        ...
