import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.stats import skellam
import pickle
import os

class XGBoostPredictor:
    def __init__(self):
        self.model_home = None
        self.model_away = None
        self.feature_columns = None

    def fit(self, training_data, feature_columns):
        """
        Trains two XGBoost Regressors (Home Goals, Away Goals) using Poisson objective.
        """
        self.feature_columns = feature_columns
        df = pd.DataFrame(training_data)
        
        # Prepare Features and Targets
        X = df[feature_columns]
        y_home = df['home_score']
        y_away = df['away_score']

        # XGBoost Configuration
        # objective='count:poisson' is ideal for goal counts
        params = {
            'objective': 'count:poisson',
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3, # Keep it simple to avoid overfitting
            'eval_metric': 'poisson-nloglik'
        }

        print("   🧠 Training XGBoost Home Model...")
        self.model_home = xgb.XGBRegressor(**params)
        self.model_home.fit(X, y_home)

        print("   🧠 Training XGBoost Away Model...")
        self.model_away = xgb.XGBRegressor(**params)
        self.model_away.fit(X, y_away)

    def predict_match_outcome(self, match_features):
        """
        Predicts match outcome using the trained models.
        Returns a dictionary with probabilities and expected scores.
        """
        if not self.model_home or not self.model_away:
            raise ValueError("Models not trained!")

        # Prepare input dataframe (single row)
        # Ensure columns match training order
        input_data = {col: [match_features.get(col, 0)] for col in self.feature_columns}
        X_input = pd.DataFrame(input_data)

        # Predict Expected Goals (Lambda)
        lambda_home = self.model_home.predict(X_input)[0]
        lambda_away = self.model_away.predict(X_input)[0]

        # Calculate Probabilities using Skellam Distribution
        # Skellam describes the difference between two Poisson variables
        prob_draw = skellam.pmf(0, lambda_home, lambda_away)
        prob_home_win = 1 - skellam.cdf(0, lambda_home, lambda_away) # P(H-A > 0)
        prob_away_win = skellam.cdf(-1, lambda_home, lambda_away)    # P(H-A < 0)

        return {
            'expected_goals_home': float(lambda_home),
            'expected_goals_away': float(lambda_away),
            'prob_home_win': float(prob_home_win),
            'prob_draw': float(prob_draw),
            'prob_away_win': float(prob_away_win)
        }

    def save_model(self, filepath):
        """Saves the predictor to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"   💾 XGBoost Model saved to {filepath}")

    def load_model(self, filepath):
        """Loads the predictor from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
            self.model_home = loaded.model_home
            self.model_away = loaded.model_away
            self.feature_columns = loaded.feature_columns
