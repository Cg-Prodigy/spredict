import json
import os
import pandas as pd
from typing import List, Dict
from datetime import datetime
import glob
import difflib
from tqdm import tqdm  # For progress bar

LEAGUE_MAP = {
    "Premier League": 17,
    "UEFA Champions League": 7,
    "UEFA Europa League": 679,
    "Laliga": 8,
    "Bundesliga": 35,
    "Serie A": 23,
    "Ligue 1": 34,
}

class DataFlattener:
    def __init__(self, json_dir: str, output_dir: str):
        self.json_dir = json_dir
        self.output_dir = output_dir
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def flatten_all_matches(self, file_pattern: str = "*.json") -> pd.DataFrame:
        """
        Reads all JSON files matching the pattern and flattens them into a single DataFrame.
        """
        json_files = glob.glob(os.path.join(self.json_dir, file_pattern))
        print(f"📂 Found {len(json_files)} match files in {self.json_dir}")
        
        all_rows = []
        
        for filepath in tqdm(json_files, desc="Processing Matches"):
            try:
                with open(filepath, 'r') as f:
                    file_content = json.load(f)
                    
                    # Handle different JSON structures
                    matches_to_process = []
                    if isinstance(file_content, dict):
                        if "data" in file_content and isinstance(file_content["data"], list):
                            # Structure: {"count": N, "data": [match1, match2, ...]}
                            matches_to_process = file_content["data"]
                        else:
                            # Structure: {match_data} (Single match file)
                            matches_to_process = [file_content]
                    elif isinstance(file_content, list):
                        # Structure: [match1, match2, ...] (List of matches)
                        matches_to_process = file_content
                        
                    for i, match_data in enumerate(matches_to_process):
                        if match_data is None:
                            print(f"⚠️ Skipping null match data at index {i} in {filepath}")
                            continue
                        try:
                            match_rows = self._process_single_match(match_data)
                            all_rows.extend(match_rows)
                        except Exception as e:
                            print(f"⚠️ Error processing match {match_data.get('match_id', 'unknown')} in {filepath}: {e}")
                        
            except Exception as e:
                print(f"⚠️ Error processing {filepath}: {e}")
                
        if not all_rows:
            print("❌ No data found.")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_rows)
        print(f"✅ Successfully flattened {len(df)} player-match records.")
        return df

    def _process_single_match(self, match_data: Dict) -> List[Dict]:
        """
        Extracts player stats from a single match dictionary.
        """
        rows = []
        
        # Match Level Info
        match_id = match_data.get("match_id")
        match_date_ts = match_data.get("match_date") # Timestamp
        
        # 1. Filter: Check for essential data
        home_formation = match_data.get("home_formation")
        away_formation = match_data.get("away_formation")
        home_score = match_data.get("home_score")
        away_score = match_data.get("away_score")
        
        # Check for missing data (None checks for scores are important as 0 is valid)
        if (not match_date_ts or 
            not home_formation or 
            not away_formation or 
            home_score is None or 
            away_score is None):
            # print(f"⚠️ Skipping match {match_id}: Missing essential data.")
            return []

        match_date = datetime.fromtimestamp(match_date_ts)
        
        # 2. Enforce integer league_id with Fuzzy Matching
        raw_league = match_data.get("league")
        league_id = 0
        
        try:
            # Try direct conversion first
            league_id = int(raw_league)
        except (ValueError, TypeError):
            # If conversion fails, try fuzzy matching against LEAGUE_MAP
            if isinstance(raw_league, str):
                matches = difflib.get_close_matches(raw_league, LEAGUE_MAP.keys(), n=1, cutoff=0.8) # 80% match threshold
                if matches:
                    matched_name = matches[0]
                    league_id = LEAGUE_MAP[matched_name]
                    # print(f"   🔄 Fuzzy Matched League: '{raw_league}' -> {matched_name} ({league_id})")
                else:
                    # print(f"   ⚠️ Could not identify league: '{raw_league}'")
                    pass
            else:
                pass
            
        home_team = match_data.get("home_team")
        away_team = match_data.get("away_team")
        home_score = match_data.get("home_score")
        away_score = match_data.get("away_score")
        
        # Process Home Players
        home_players = match_data.get("home_players")
        if home_players:
            for p in home_players:
                if p is None: continue
                rows.append(self._extract_player_row(p, match_id, match_date, league_id, 
                                                   home_team, away_team, is_home=True, 
                                                   team_score=home_score, opp_score=away_score,
                                                   formation=home_formation, opp_formation=away_formation))
            
        # Process Away Players
        away_players = match_data.get("away_players")
        if away_players:
            for p in away_players:
                if p is None: continue
                rows.append(self._extract_player_row(p, match_id, match_date, league_id, 
                                                   away_team, home_team, is_home=False, 
                                                   team_score=away_score, opp_score=home_score,
                                                   formation=away_formation, opp_formation=home_formation))
            
        return rows

    def _extract_player_row(self, player: Dict, match_id, match_date, league_id, 
                          team_name, opp_team_name, is_home, team_score, opp_score,
                          formation=None, opp_formation=None) -> Dict:
        """
        Flattens a single player's data into a dictionary row.
        """
        stats = player.get("statistics", {})
        
        row = {
            "match_id": match_id,
            "date": match_date,
            "league_id": league_id,
            "home_team": team_name if is_home else opp_team_name,
            "away_team": opp_team_name if is_home else team_name,
            "player_id": player.get("player_id"),
            "player_name": player.get("player_name"),
            "team": team_name,
            "opponent": opp_team_name,
            "is_home": 1 if is_home else 0,
            "formation": formation,
            "opp_formation": opp_formation,
            "position": player.get("position"),
            "specific_position": player.get("specific_position"),
            "minutes_played": player.get("minutes_played"),
            "rating": player.get("rating"),
            "substitute": 1 if player.get("substitute") else 0,
            "goals": stats.get("goals", 0),
            "assists": stats.get("goalAssist", 0),
            "total_shots": stats.get("totalShots", 0),
            "shots_on_target": stats.get("onTargetScoringAttempt", 0),
            "total_passes": stats.get("totalPass", 0),
            "passes_completed": stats.get("accuratePass", 0),
            "total_tackles": stats.get("totalTackle", 0),
            "tackles_won": stats.get("wonTackle", 0),
            "interceptions": stats.get("interceptionWon", 0),
            "duels_won": stats.get("duelWon", 0),
            "possession_lost": stats.get("possessionLostCtrl", 0),
            "fouls_conceded": stats.get("fouls", 0),
            "yellow_cards": 1 if stats.get("yellowCard") else 0,
            "red_cards": 1 if stats.get("redCard") else 0,
            "team_score": team_score,
            "opp_score": opp_score,
            "expected_assists": stats.get("expectedAssists", 0),
            "big_chance_missed": stats.get("bigChanceMissed", 0),
            "tackle_success": stats.get("wonTackle", 0) / (stats.get("totalTackle", 0) + 1e-6), # Avoid div by zero
            "dribble_success": stats.get("wonContest", 0) / (stats.get("totalContest", 0) + 1e-6),
            "won_contest": stats.get("wonContest", 0),
            "total_contest": stats.get("totalContest", 0),
            "penalty_won": stats.get("penaltyWon", 0),
            "penalty_conceded": stats.get("penaltyConceded", 0),
            "saves": stats.get("saves", 0),
            "errors_leading_to_goal": stats.get("errorLeadToGoal", 0)
        }
        return row

    def save_dataframe(self, df: pd.DataFrame, filename: str = "master_player_stats.csv"):
        """
        Saves the DataFrame to CSV (and Parquet if available).
        """
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"💾 Saved Master DataFrame to: {output_path}")
        
        # Optional: Save as Parquet for speed if pyarrow/fastparquet is installed
        try:
            parquet_path = output_path.replace(".csv", ".parquet")
            df.to_parquet(parquet_path, index=False)
            print(f"💾 Saved Parquet version to: {parquet_path}")
        except ImportError:
            print("ℹ️ Parquet library not found, skipping parquet save.")
        except Exception as e:
            print(f"⚠️ Could not save parquet: {e}")

if __name__ == "__main__":
    # Configuration
    # Adjust these paths to match your actual directory structure
    JSON_DIR = "json_data/league_info/" # Change this to your actual data folder, e.g., "json_data/completed_stats"
    OUTPUT_DIR = "data"
    
    flattener = DataFlattener(JSON_DIR, OUTPUT_DIR)
    
    # Run
    # Note: In production, you might want to change "*.json" to match your specific file naming convention
    df = flattener.flatten_all_matches("*") 
    
    if not df.empty:
        flattener.save_dataframe(df)
        print("\nFirst 5 rows of the Master DataFrame:")
        print(df.head())
