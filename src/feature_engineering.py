"""
Loads the fitted transformers saved by Notebook 5 (imputers, scaler, encoder)
and applies them to new data. These objects are NEVER re-fitted here --
only loaded and used with .transform(), exactly as the task requires.
"""

import logging
import numpy as np
import pandas as pd
import joblib

from src.config_loader import load_config, resolve_path

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Wraps the saved transformers and turns a raw order DataFrame into a feature vector."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        paths = self.config["paths"]

        self.num_imputer = joblib.load(resolve_path(paths["num_imputer"]))
        self.scaler = joblib.load(resolve_path(paths["scaler"]))
        self.cat_imputer = joblib.load(resolve_path(paths["cat_imputer"]))
        self.encoder = joblib.load(resolve_path(paths["encoder"]))

        self.numeric_features = self.config["features"]["numeric_features"]
        self.categorical_features = self.config["features"]["categorical_features"]

        with open(resolve_path(paths["feature_list"])) as f:
            self.feature_names = [line.strip() for line in f if line.strip()]

        logger.info("Loaded all fitted transformers and feature list.")

    def transform(self, df):
        """Turn a DataFrame with raw order fields into the final feature matrix,
        returned as a DataFrame with the exact column names the model was trained on
        (avoids the sklearn 'X does not have valid feature names' warning)."""
        num = self.num_imputer.transform(df[self.numeric_features])
        num = self.scaler.transform(num)

        cat = self.cat_imputer.transform(df[self.categorical_features])
        cat = self.encoder.transform(cat)

        X = np.hstack([num, cat])
        X_df = pd.DataFrame(X, columns=self.feature_names)
        return X_df