import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import skellam
from typing import Dict, List, Tuple

class MatchPredictor:
    def __init__(self):
        self.model_home = None
        self.model_away = None
        self.scaler = None
        
    def prepare_match_features(self, current_match: Dict, aws_calculator, interaction_model, use_interactions=True) -> Dict:
        """
        Prepare features for a single match prediction.
        """
        home_players = current_match['home_data']
        away_players = current_match['away_data']
        
        # Calculate AWS for all players first
        # We need to pass the match date to the calculator!
        match_date = current_match.get('start_time') # Timestamp
        if match_date:
             # If it's a timestamp, convert to datetime? 
             # The calculator expects a dictionary with 'start_time' or we can pass it manually?
             # The calculator's calculate_aws_for_match takes match_data dict.
             # Let's assume we call calculate_aws_for_match separately or here.
             # Actually, the players might already have 'aws' if we ran the calculator before.
             # But let's assume we need to calculate it if missing.
             pass

        # Check if players have AWS ratings, if not, calculate them?
        # For now, assume 'aws' key exists in player dicts (populated by MatchAWSBuilder)
        
        # Calculate Δ_start
        if use_interactions and interaction_model:
            delta_start = interaction_model.calculate_delta_start(
                home_players, 
                away_players, 
                current_match.get('detected_home_formation', '4-4-2'), 
                current_match.get('detected_away_formation', '4-4-2')
            )
        else:
            # Fallback to simple average
            home_starters = [p for p in home_players if not p.get('substitute', False)]
            away_starters = [p for p in away_players if not p.get('substitute', False)]
            delta_start = (
                np.mean([p.get('aws', 0) for p in home_starters]) - 
                np.mean([p.get('aws', 0) for p in away_starters])
            ) if home_starters and away_starters else 0.0
        
        # Calculate Δ_sub
        home_subs = [p for p in home_players if p.get('substitute', False)]
        away_subs = [p for p in away_players if p.get('substitute', False)]
        delta_sub = (
            np.mean([p.get('aws', 0) for p in home_subs]) - 
            np.mean([p.get('aws', 0) for p in away_subs])
        ) if home_subs and away_subs else 0.0
        
        return {
            'match_id': current_match.get('match_id'),
            'home_team': current_match['home_team'],
            'away_team': current_match['away_team'],
            'Δ_start': delta_start,
            'Δ_sub': delta_sub,
            'home_formation': current_match.get('detected_home_formation'),
            'away_formation': current_match.get('detected_away_formation'),
            'used_interactions': use_interactions
        }

    def fit_skellam_model(self, training_data: List[Dict], feature_columns: List[str] = None):
        """
        Fit the Skellam model (via two Poisson models) on historical data.
        """
        df = pd.DataFrame(training_data)
        
        # Features
        if feature_columns:
            feature_cols = feature_columns
        else:
            # Default Standard Features
            event_features = ['Δ_pass_accuracy', 'Δ_tackles', 'Δ_duels', 'Δ_shots_on_target', 'Δ_possession_lost', 'Δ_shot_accuracy', 'Δ_conversion_rate']
            feature_cols = ['Δ_start', 'Δ_sub'] + event_features
        
        X = df[feature_cols]
        X = sm.add_constant(X)
        
        # Targets
        y_home = df['home_score']
        y_away = df['away_score']
        
        # Fit Poisson models (GLM)
        # Now that data is clean (no NaNs in targets), this should work reliably
        print("🧠 Fitting GLM Poisson Model...")
        self.model_home = sm.GLM(y_home, X, family=sm.families.Poisson()).fit()
        self.model_away = sm.GLM(y_away, X, family=sm.families.Poisson()).fit()
        
        print("✅ Skellam Model Trained")
        print(self.model_home.summary())

    def predict_match_outcome(self, match_features: Dict) -> Dict:
        """
        Predict outcome probabilities for a single match.
        """
        if not self.model_home or not self.model_away:
            raise ValueError("Model must be fitted before prediction")
            
        # Prepare input
        
        # Prepare input dynamically based on model features
        # The model params include 'const' and feature names
        feature_names = self.model_home.params.index.tolist()
        
        input_data = {'const': 1.0}
        for feat in feature_names:
            if feat == 'const': continue
            # Get feature from match_features, default to 0 if missing
            input_data[feat] = match_features.get(feat, 0)
        
        X = pd.DataFrame([input_data])
        
        # Predict expected goals (lambda)
        mu_home = self.model_home.predict(X)[0]
        mu_away = self.model_away.predict(X)[0]
        
        # Calculate probabilities using Skellam distribution
        # P(Home Win) = P(Home - Away > 0)
        # P(Draw) = P(Home - Away = 0)
        # P(Away Win) = P(Home - Away < 0)
        
        prob_home_win = 0.0
        prob_draw = 0.0
        prob_away_win = 0.0
        
        # Sum probabilities for a reasonable range of goal differences
        for k in range(-10, 11):
            prob = skellam.pmf(k, mu_home, mu_away)
            if k > 0:
                prob_home_win += prob
            elif k == 0:
                prob_draw += prob
            else:
                prob_away_win += prob
                
        return {
            'home_team': match_features['home_team'],
            'away_team': match_features['away_team'],
            'prob_home_win': prob_home_win,
            'prob_draw': prob_draw,
            'prob_away_win': prob_away_win,
            'expected_goals_home': mu_home,
            'expected_goals_away': mu_away
        }

    def save_model(self, filepath: str):
        """
        Save the trained model to a file.
        """
        import pickle
        if not self.model_home or not self.model_away:
            raise ValueError("Cannot save: Model is not fitted.")
            
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model_home': self.model_home,
                'model_away': self.model_away
            }, f)
        print(f"💾 Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load a trained model from a file.
        """
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model_home = data['model_home']
            self.model_away = data['model_away']
        print(f"📂 Model loaded from {filepath}")
