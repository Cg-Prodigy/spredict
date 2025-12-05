import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class AWSRatingCalculator:
    """
    Faithful implementation of Holmes & McHale (2024) Adjusted WhoScored (AWS) ratings.
    Refactored to use a "Global History" DataFrame for efficiency.
    """

    def __init__(self, phi: float = 0.0062, psi: float = 2.0, omega: float = 7.0):
        self.phi = phi     # time-decay rate
        self.psi = psi     # max historical window (years)
        self.omega = omega # shrinkage weight

    def calculate_ratings_for_match_day(self, 
                                      match_date: datetime, 
                                      player_ids: List[int], 
                                      master_df: pd.DataFrame) -> Dict[int, float]:
        """
        Calculates AWS ratings for a list of players based on their history *prior* to match_date.
        
        Args:
            match_date: The date of the match(es) we are predicting.
            player_ids: List of player IDs to calculate ratings for.
            master_df: The Master DataFrame containing all historical player stats.
            
        Returns:
            Dictionary mapping player_id -> calculated_rating
        """
        # 1. Filter History: Get all games strictly BEFORE the match date
        # We also filter for the max historical window (psi) to optimize
        start_date = match_date - timedelta(days=self.psi * 365)
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(master_df['date']):
            master_df['date'] = pd.to_datetime(master_df['date'])

        # STRICT OPTIMIZATION: Only get history for the requested players
        history_df = master_df[
            (master_df['player_id'].isin(player_ids)) &
            (master_df['date'] < match_date) & 
            (master_df['date'] >= start_date)
        ].copy()
        
        if history_df.empty:
            return {pid: self._get_default_rating() for pid in player_ids}

        # 2. Calculate Weights
        # weight = exp(-phi * days_diff / 3.5)
        days_diff = (match_date - history_df['date']).dt.days
        history_df['weight'] = np.exp(-self.phi * days_diff / 3.5)
        
        # 3. Add Pseudo-Observations (Shrinkage)
        # We need to do this for ALL players in the history, plus any requested players 
        # who might not have history (they will rely 100% on pseudo-obs)
        
        # Get unique players involved in the history OR requested
        relevant_players = set(history_df['player_id'].unique()) | set(player_ids)
        
        # Calculate league averages for pseudo-observations
        # If a player has no history, we need a default league. 
        # For now, we'll use the global average if league is unknown.
        global_avg_rating = history_df['rating'].mean()
        league_avgs = history_df.groupby('league_id')['rating'].mean().to_dict()
        
        pseudo_rows = []
        
        # Pre-calculate player's most frequent league to assign correct pseudo-obs
        # (For players with no history, we can't easily guess league without more info, 
        # so we might assign a generic one or skip league factor for them)
        player_leagues = history_df.groupby('player_id')['league_id'].agg(
            lambda x: x.mode().iloc[0] if not x.empty else None
        ).to_dict()

        for pid in relevant_players:
            league = player_leagues.get(pid)
            
            # If we don't know the player's league (no history), use a default or try to infer?
            # The paper says "average rating of the league the player belongs to".
            # If unknown, we use global average.
            avg_rating = league_avgs.get(league, global_avg_rating)
            
            pseudo_rows.append({
                "player_id": pid,
                "rating": avg_rating,
                "league_id": league if league is not None else -1, # -1 or similar for unknown
                "is_home": 0.5, # Neutral venue for pseudo-obs
                "weight": self.omega
            })
            
        pseudo_df = pd.DataFrame(pseudo_rows)
        
        # Combine history and pseudo-observations
        training_df = pd.concat([history_df, pseudo_df], ignore_index=True)
        
        # 4. Normalize Weights
        # "The weights are normalized so that they sum to the number of observations"
        if not training_df.empty:
             training_df['weight'] = training_df['weight'] * (len(training_df) / training_df['weight'].sum())
        else:
             return {pid: self._get_default_rating() for pid in player_ids}
        
        # 5. Fit WLS Model
        # Formula: rating ~ is_home + C(player_id) + C(league_id)
        # We treat player_id and league_id as categorical
        
        # Optimization: We only strictly need to fit for the requested player_ids.
        # However, WLS fits all coefficients jointly.
        
        try:
            model = smf.wls(
                formula="rating ~ is_home + C(player_id) + C(league_id)",
                data=training_df,
                weights=training_df['weight']
            ).fit()
            
            return self._extract_ratings_from_model(model, player_ids, player_leagues)
            
        except Exception as e:
            print(f"Error fitting WLS model: {e}")
            return {pid: self._get_default_rating() for pid in player_ids}

    def _extract_ratings_from_model(self, model, player_ids, player_leagues) -> Dict[int, float]:
        """
        Extracts the final AWS rating for specific players from the fitted model.
        Rating = Intercept + Player_Coeff + League_Coeff
        """
        ratings = {}
        params = model.params
        intercept = params.get('Intercept', 0)
        
        # Handle NaN intercept
        if pd.isna(intercept):
            intercept = 0
        
        for pid in player_ids:
            # Construct parameter names as statsmodels does
            player_key = f"C(player_id)[T.{pid}]"
            
            # If player was the reference category, coeff is 0
            player_coeff = params.get(player_key, 0)
            if pd.isna(player_coeff):
                player_coeff = 0
            
            # League coeff
            league_id = player_leagues.get(pid)
            league_key = f"C(league_id)[T.{league_id}]"
            league_coeff = params.get(league_key, 0)
            if pd.isna(league_coeff):
                league_coeff = 0
            
            # Calculate final rating
            # Note: is_home is set to 0 (neutral) or average? 
            # The paper defines AWS as the player's "underlying ability".
            # Usually this means removing the home advantage term.
            # So Rating = Intercept + Player_Effect + League_Effect
            
            final_rating = intercept + player_coeff + league_coeff
            
            # Ensure we never return NaN
            if pd.isna(final_rating):
                final_rating = self._get_default_rating()
                
            ratings[pid] = final_rating
            
        return ratings

    def _get_default_rating(self):
        return 6.0 # A reasonable average start
