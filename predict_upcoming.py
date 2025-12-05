import argparse
import json
import glob
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aws_calculator import AWSRatingCalculator
from match_predictor import MatchPredictor
from interaction_model import InteractionModel
from match_collector import MatchCollector
from xgboost_model import XGBoostPredictor

def main():
    parser = argparse.ArgumentParser(description="Predict matches for a specific time.")
    parser.add_argument("time", help="Match start time (HH:MM), e.g., '23:00'")
    args = parser.parse_args()
    
    target_time = args.time
    
    print(f"🚀 Starting Prediction for matches at {target_time}...")

    # 1. Find latest games file
    # We are running from src/, so json_data is in current dir
    files = glob.glob("json_data/games_today/*.json")
    # print(f"DEBUG: Found files: {files}")
    if not files:
        print("❌ No games data found. Run get_todays_games.py first.")
        return
    # Sort by date in filename (YYYY-MM-DD)
    latest_file = max(files, key=lambda x: os.path.basename(x).split('-games')[0])
    print(f"📂 Loading games from: {latest_file}")
    
    with open(latest_file) as f:
        games_groups = json.load(f)
        
    if target_time not in games_groups:
        print(f"❌ No games found for time {target_time}. Available times: {list(games_groups.keys())}")
        return
        
    matches = games_groups[target_time]
    print(f"🔍 Found {len(matches)} matches at {target_time}")
    
    # 2. Load Models & Data
    print("⚙️ Loading Models and Data...")
    try:
        aws_calc = AWSRatingCalculator()
        interaction_model = InteractionModel()
        
        # Load Standard Model
        predictor_std = MatchPredictor()
        predictor_std.load_model("models/skellam_model_avg.pkl")
        
        # Load Tier 1 Model (if exists)
        predictor_t1 = MatchPredictor()
        t1_path = "models/skellam_model_tier1.pkl"
        has_t1_model = os.path.exists(t1_path)
        if has_t1_model:
            predictor_t1.load_model(t1_path)
            # print("   💎 Tier 1 Model Loaded")
        else:
            # print("   ⚠️ Tier 1 Model not found. Using Standard Model only.")
            pass
            
        # Load XGBoost Model (Experimental)
        predictor_xgb = XGBoostPredictor()
        xgb_path = "models/xgboost_model.pkl"
        has_xgb_model = os.path.exists(xgb_path)
        if has_xgb_model:
            predictor_xgb.load_model(xgb_path)
            # print("   🚀 XGBoost Model Loaded")
        else:
            pass
        
        # print("   🔍 Standard Model Parameters:")
        # print(f"   Home Model: {predictor_std.model_home.params.to_dict()}")
        # print(f"   Away Model: {predictor_std.model_away.params.to_dict()}")
        
        master_df = pd.read_csv("data/master_player_stats.csv")
        master_df['date'] = pd.to_datetime(master_df['date'])
    except Exception as e:
        print(f"❌ Error loading models/data: {e}")
        return

    # Load Master Data for Form Calculation
    print("📂 Loading Master Data for Form Calculation...")
    form_master_df = pd.read_csv("data/master_player_stats.csv", low_memory=False)
    # Enforce integer league_id
    if 'league_id' in form_master_df.columns:
        form_master_df['league_id'] = pd.to_numeric(form_master_df['league_id'], errors='coerce').fillna(0).astype(int)
        
    form_master_df['date'] = pd.to_datetime(form_master_df['date'])
    form_master_df = form_master_df.sort_values('date')
    
    
    # Pre-calculate form for all teams
    team_history = {}
    team_event_history = {}
    
    # Iterate through all matches to build history
    # We need to group by match_id to get results
    match_groups = form_master_df.groupby('match_id')
    for mid, group in match_groups:
        # Ensure group is not empty before accessing elements
        if group.empty:
            continue

        # Get unique team names for the match
        home_team_row = group[group['is_home'] == 1]
        away_team_row = group[group['is_home'] == 0]

        if home_team_row.empty or away_team_row.empty:
            continue # Skip if home or away team data is missing for this match

        h_team = home_team_row.iloc[0]['team']
        a_team = away_team_row.iloc[0]['team']
        
        # Determine score
        # The master csv has 'team_score' and 'opponent_score'
        # For home team: team_score is home_score
        h_score = home_team_row.iloc[0]['team_score']
        a_score = home_team_row.iloc[0]['opp_score'] # opponent_score from home team perspective is away team's score
        
        # Determine points (Standard 3/1/0)
        if h_score > a_score:
            h_pts, a_pts = 3, 0
        elif h_score == a_score:
            h_pts, a_pts = 1, 1
        else:
            h_pts, a_pts = 0, 3
            
        if h_team not in team_history: team_history[h_team] = []
        if a_team not in team_history: team_history[a_team] = []
        
        team_history[h_team].append(h_pts)
        team_history[a_team].append(a_pts)
        
        # Update Event History
        # Calculate team stats from group
        home_rows = group[group['is_home'] == 1]
        away_rows = group[group['is_home'] == 0]
        
        def get_team_stats(rows):
            total_passes = rows['total_passes'].sum()
            accurate_passes = rows['passes_completed'].sum()
            pass_acc = accurate_passes / total_passes if total_passes > 0 else 0
            
            total_shots = rows['total_shots'].sum()
            shots_on_target = rows['shots_on_target'].sum()
            goals = rows['goals'].sum()
            
            shot_acc = shots_on_target / total_shots if total_shots > 0 else 0
            conversion = goals / shots_on_target if shots_on_target > 0 else 0
            
            # Tier 1 Stats Calculation
            total_tackles = rows['total_tackles'].sum()
            won_tackles = rows['tackles_won'].sum()
            tackle_success = won_tackles / total_tackles if total_tackles > 0 else 0
            
            total_contests = rows['total_contest'].sum()
            won_contests = rows['won_contest'].sum()
            dribble_success = won_contests / total_contests if total_contests > 0 else 0
            
            return {
                'pass_accuracy': pass_acc,
                'tackles_won': won_tackles,
                'duels_won': rows['duels_won'].sum(),
                'shots_on_target': shots_on_target,
                'possession_lost': rows['possession_lost'].sum(),
                'shot_accuracy': shot_acc,
                'conversion_rate': conversion,
                'expected_assists': rows['expected_assists'].sum(),
                'big_chance_missed': rows['big_chance_missed'].sum(),
                'tackle_success': tackle_success,
                'interceptions': rows['interceptions'].sum(),
                'dribble_success': dribble_success
            }

        h_stats = get_team_stats(home_rows)
        a_stats = get_team_stats(away_rows)
        
        if h_team not in team_event_history: team_event_history[h_team] = []
        if a_team not in team_event_history: team_event_history[a_team] = []
        
        team_event_history[h_team].append(h_stats)
        team_event_history[a_team].append(a_stats)
        
    def get_current_form(team_name):
        if team_name not in team_history:
            return 5.0 # Default form
        
        # Calculate weighted average of last 5 games
        recent_games = team_history[team_name][-5:]
        if not recent_games: return 5.0
        
        weights = np.exp(np.linspace(-1, 0, len(recent_games)))
        return np.average(recent_games, weights=weights)

    def get_event_form(team_name, history_dict, metric):
        if team_name not in history_dict:
            return 0.0
        
        recent = history_dict[team_name][-5:]
        if not recent: return 0.0
        
        # Extract metric values
        values = [g[metric] for g in recent if metric in g] # Ensure metric exists
        if not values: return 0.0
        
        weights = np.exp(np.linspace(-0.5, 0, len(values))) # Less decay for stats
        return np.average(values, weights=weights)

    # 3. Process Matches
    TIER1_LEAGUES = [7, 17, 8, 23, 35, 34, 679] # Example IDs for Premier League, La Liga, etc.
    
    for match in matches:
        print(f"\n--------------------------------------------------")
        print(f"⚽ {match['home team']} vs {match['away team']}")
        print(f"--------------------------------------------------")
        
        # Fetch Lineup
        try:
            print("   📡 Fetching Lineup...")
            collector = MatchCollector(match)
            collector.getLineUp() # Updates collector.complete_match_obj
            
            full_match_data = collector.complete_match_obj
            
            if "home_players" not in full_match_data:
                 print("   ⚠️ Lineup not confirmed/found. Skipping.")
                 continue
            
            # Determine League ID (heuristic or from data)
            # match object usually has 'tournament' -> 'uniqueTournament' -> 'id'
            league_id = match.get('tournament', {}).get('uniqueTournament', {}).get('id')
            if not league_id:
                league_id = match.get('tournament_id')
            
            # Check if Tier 1
            is_tier1 = league_id in TIER1_LEAGUES and has_t1_model
            model_to_use = predictor_t1 if is_tier1 else predictor_std
            model_name = "💎 Tier 1" if is_tier1 else "⚖️ Standard"
            
            print(f"   ⚙️ Using Model: {model_name}")

            home_players_raw = full_match_data["home_players"]
            away_players_raw = full_match_data["away_players"]
            
            # Extract Player IDs
            home_ids = [p['player_id'] for p in home_players_raw]
            away_ids = [p['player_id'] for p in away_players_raw]
            all_ids = home_ids + away_ids
            
            # Calculate Current Ratings (using history up to TODAY)
            match_date = datetime.fromtimestamp(match['start_time'])
            
            print(f"   📊 Calculating Ratings for {len(all_ids)} players...")
            ratings = aws_calc.calculate_ratings_for_match_day(match_date, all_ids, master_df)
            
            # Prepare Player Objects
            home_objs = []
            for p in home_players_raw:
                pid = p['player_id']
                home_objs.append({
                    'id': pid, 'name': p['player_name'], 'standard_position': p.get('position', 'CM'),
                    'aws': ratings.get(pid, 6.0), 'substitute': p.get('substitute', False)
                })
                
            away_objs = []
            for p in away_players_raw:
                pid = p['player_id']
                away_objs.append({
                    'id': pid, 'name': p['player_name'], 'standard_position': p.get('position', 'CM'),
                    'aws': ratings.get(pid, 6.0), 'substitute': p.get('substitute', False)
                })
                
            # Formations
            h_fmt = full_match_data.get("home_formation", "4-4-2")
            a_fmt = full_match_data.get("away_formation", "4-4-2")
            
            # Calculate Features
            delta_start = interaction_model.calculate_delta_start(home_objs, away_objs, h_fmt, a_fmt)
            
            home_subs = [p for p in home_objs if p['substitute']]
            away_subs = [p for p in away_objs if p['substitute']]
            avg_h_sub = np.mean([p['aws'] for p in home_subs]) if home_subs else 0
            avg_a_sub = np.mean([p['aws'] for p in away_subs]) if away_subs else 0
            delta_sub = avg_h_sub - avg_a_sub
            
            # Calculate Event Form
            # Standard Metrics
            metrics = ['pass_accuracy', 'tackles_won', 'duels_won', 'shots_on_target', 'possession_lost', 'shot_accuracy', 'conversion_rate']
            
            # Tier 1 Metrics (if applicable)
            if is_tier1:
                metrics += ['expected_assists', 'big_chance_missed', 'tackle_success', 'interceptions', 'dribble_success']
            
            h_ev = {m: get_event_form(match['home team'], team_event_history, m) for m in metrics}
            a_ev = {m: get_event_form(match['away team'], team_event_history, m) for m in metrics}
            
            print(f"   📈 Features: Δ_start={delta_start:.2f}, Δ_sub={delta_sub:.2f}")
            
            # Build Feature Dict
            features = {
                'home_team': match['home team'],
                'away_team': match['away team'],
                'Δ_start': delta_start,
                'Δ_sub': delta_sub,
                'Δ_pass_accuracy': h_ev['pass_accuracy'] - a_ev['pass_accuracy'],
                'Δ_tackles': h_ev['tackles_won'] - a_ev['tackles_won'],
                'Δ_duels': h_ev['duels_won'] - a_ev['duels_won'],
                'Δ_shots_on_target': h_ev['shots_on_target'] - a_ev['shots_on_target'],
                'Δ_possession_lost': h_ev['possession_lost'] - a_ev['possession_lost'],
                'Δ_shot_accuracy': h_ev['shot_accuracy'] - a_ev['shot_accuracy'],
                'Δ_conversion_rate': h_ev['conversion_rate'] - a_ev['conversion_rate']
            }
            
            # --- Prediction Logic ---
            
            # 1. Standard Model Prediction (PRIMARY)
            pred_std = predictor_std.predict_match_outcome(features)
            
            print(f"   ⚖️ Standard Model (Primary):")
            print(f"      🏠 {match['home team']}: {pred_std['prob_home_win']:.1%} | 🤝 Draw: {pred_std['prob_draw']:.1%} | ✈️ {match['away team']}: {pred_std['prob_away_win']:.1%}")
            print(f"      Score: {pred_std['expected_goals_home']:.2f} - {pred_std['expected_goals_away']:.2f}")

            # Set Final Prediction to Standard Model (User Preference)
            final_pred = pred_std 

            # 2. Tier 1 Model Prediction (Supplementary)
            if is_tier1:
                # Add Tier 1 specific features
                features_t1 = features.copy()
                features_t1.update({
                    'Δ_expected_assists': h_ev.get('expected_assists', 0) - a_ev.get('expected_assists', 0),
                    'Δ_big_chance_missed': h_ev.get('big_chance_missed', 0) - a_ev.get('big_chance_missed', 0),
                    'Δ_tackle_success': h_ev.get('tackle_success', 0) - a_ev.get('tackle_success', 0),
                    'Δ_interceptions': h_ev.get('interceptions', 0) - a_ev.get('interceptions', 0),
                    'Δ_dribble_success': h_ev.get('dribble_success', 0) - a_ev.get('dribble_success', 0)
                })
                
                pred_t1 = predictor_t1.predict_match_outcome(features_t1)
                
                print(f"   💎 Tier 1 Model (Analyst View):")
                print(f"      🏠 {match['home team']}: {pred_t1['prob_home_win']:.1%} | 🤝 Draw: {pred_t1['prob_draw']:.1%} | ✈️ {match['away team']}: {pred_t1['prob_away_win']:.1%}")
                print(f"      Score: {pred_t1['expected_goals_home']:.2f} - {pred_t1['expected_goals_away']:.2f}")
                
                # NOTE: We do NOT update final_pred here. We stick with Standard.

            # 3. XGBoost Model Prediction (Experimental)
            if has_xgb_model and is_tier1:
                # Use Tier 1 features for XGBoost as trained
                pred_xgb = predictor_xgb.predict_match_outcome(features_t1)
                
                print(f"   🚀 XGBoost Model (AI Experimental):")
                print(f"      🏠 {match['home team']}: {pred_xgb['prob_home_win']:.1%} | 🤝 Draw: {pred_xgb['prob_draw']:.1%} | ✈️ {match['away team']}: {pred_xgb['prob_away_win']:.1%}")
                print(f"      Score: {pred_xgb['expected_goals_home']:.2f} - {pred_xgb['expected_goals_away']:.2f}")

            # Determine Advice/Confidence based on STANDARD MODEL
            p_home = final_pred['prob_home_win']
            p_away = final_pred['prob_away_win']
            p_draw = final_pred['prob_draw']
            max_prob = max(p_home, p_away, p_draw)
            
            advice = ""
            if max_prob > 0.70:
                advice = "🔥 BANKER (High Confidence)"
            elif abs(p_home - p_away) < 0.05 and max_prob > p_draw:
                advice = "💎 VALUE PICK (Tight Call)"
            elif p_away > p_home and p_away < 0.60:
                advice = "⚠️ RISKY (Away Favorite < 60%)"
            else:
                advice = "⚖️ STANDARD"

            # Explicit Winner
            if p_home > p_away and p_home > p_draw:
                winner = f"🏠 {match['home team']}"
            elif p_away > p_home and p_away > p_draw:
                winner = f"✈️ {match['away team']}"
            else:
                winner = "🤝 DRAW"

            print(f"   🔮 Advice: {advice}")
            print(f"   🏆 PREDICTED WINNER: {winner} ({max_prob:.1%})")
            
        except Exception as e:
            print(f"   ❌ Error fetching/predicting: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
