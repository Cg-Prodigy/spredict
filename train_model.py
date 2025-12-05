import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aws_calculator import AWSRatingCalculator
from match_predictor import MatchPredictor
# Assuming InteractionModel is in interaction_model.py
from interaction_model import InteractionModel 
from xgboost_model import XGBoostPredictor 

def train_pipeline():
    print("🚀 Starting Full Model Training Pipeline...")
    
    # 1. Load Data
    data_path = "src/data/master_player_stats.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return

    print("📂 Loading Master DataFrame...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date to ensure we process chronologically (past -> future)
    df = df.sort_values('date')
    
    # Group by match
    match_ids = df['match_id'].unique()
    print(f"📊 Found {len(match_ids)} matches to process.")
    
    # 2. Initialize Components
    aws_calc = AWSRatingCalculator()
    interaction_model = InteractionModel() # Ensure this class is available
    predictor = MatchPredictor()
    
    training_data = []
    team_history = {} # Stores list of points for each team
    
    
    # 3. Feature Generation Loop
    print("⚙️ Generating Features (Ratings & Interactions)...")
    
    # We can optimize by batching, but for clarity/correctness let's loop matches first.
    # To speed up AWS, we might want to cache ratings or update them daily.
    # For this script, we'll call the calculator for each match.
    
    success_count = 0
    error_count = 0
    
    team_event_history = {} # {team: [{stats}, ...]}

    for mid in tqdm(match_ids):
        try:
            match_rows = df[df['match_id'] == mid]
            if match_rows.empty:
                continue
                
            # Basic Match Info
            match_date = match_rows.iloc[0]['date']
            home_team = match_rows[match_rows['is_home'] == 1].iloc[0]['team']
            away_team = match_rows[match_rows['is_home'] == 0].iloc[0]['team']
            home_score = match_rows.iloc[0]['team_score'] if match_rows.iloc[0]['is_home'] == 1 else match_rows.iloc[0]['opp_score']
            away_score = match_rows.iloc[0]['opp_score'] if match_rows.iloc[0]['is_home'] == 1 else match_rows.iloc[0]['team_score']
            
            # Get Players
            home_players_ids = match_rows[match_rows['is_home'] == 1]['player_id'].tolist()
            away_players_ids = match_rows[match_rows['is_home'] == 0]['player_id'].tolist()
            all_players = home_players_ids + away_players_ids
            
            # Calculate Pre-Match Ratings
            # Note: This uses history STRICTLY BEFORE match_date
            ratings = aws_calc.calculate_ratings_for_match_day(match_date, all_players, df)
            
            # Construct Player Objects for Interaction Model
            # We need to map the flat rows back to objects with 'position', 'aws', etc.
            
            home_player_objs = []
            for _, row in match_rows[match_rows['is_home'] == 1].iterrows():
                pid = row['player_id']
                home_player_objs.append({
                    'id': pid,
                    'name': row['player_name'],
                    'position': row['position'], # e.g. 'GK', 'CB'
                    'specific_position': row['specific_position'], # e.g. 'RCB'
                    'aws': ratings.get(pid, 6.0), # The calculated rating
                    'substitute': row['substitute'] == 1
                })
                
            away_player_objs = []
            for _, row in match_rows[match_rows['is_home'] == 0].iterrows():
                pid = row['player_id']
                away_player_objs.append({
                    'id': pid,
                    'name': row['player_name'],
                    'position': row['position'],
                    'specific_position': row['specific_position'],
                    'aws': ratings.get(pid, 6.0),
                    'substitute': row['substitute'] == 1
                })
            
            # Determine Formations from Data
            # We take the first non-null formation found for each team
            home_fmt = "4-4-2"
            away_fmt = "4-4-2"
            
            home_rows = match_rows[match_rows['is_home'] == 1]
            away_rows = match_rows[match_rows['is_home'] == 0]
            
            if not home_rows.empty and 'formation' in home_rows.columns:
                fmt = home_rows['formation'].iloc[0]
                if pd.notna(fmt): home_fmt = str(fmt)
                
            if not away_rows.empty and 'formation' in away_rows.columns:
                fmt = away_rows['formation'].iloc[0]
                if pd.notna(fmt): away_fmt = str(fmt)
            
            # Calculate Delta Start (Interaction Weighted)
            delta_start = interaction_model.calculate_delta_start(
                home_player_objs, 
                away_player_objs, 
                home_fmt, 
                away_fmt
            )
            
            # Calculate Delta Sub (Simple Average Difference)
            home_subs = [p for p in home_player_objs if p['substitute']]
            away_subs = [p for p in away_player_objs if p['substitute']]
            
            avg_home_sub = np.mean([p['aws'] for p in home_subs]) if home_subs else 0
            avg_away_sub = np.mean([p['aws'] for p in away_subs]) if away_subs else 0
            delta_sub = avg_home_sub - avg_away_sub
            
            # Calculate Recent Form (Last 5 Games Points)
            # team_history is a dict: {team_name: [points, ...]}
            # We need to initialize this outside the loop
            
            def get_form(team_name, history):
                if team_name not in history:
                    return 0
                
                # Get last 5 games
                last_5 = history[team_name][-5:]
                
                # Apply exponential decay
                # Most recent game gets weight 1.0, previous 0.85, etc.
                decay_factor = 0.85
                weighted_score = 0
                
                # Iterate backwards (newest to oldest)
                for i, points in enumerate(reversed(last_5)):
                    weight = decay_factor ** i
                    weighted_score += points * weight
                    
                return weighted_score

            def get_event_form(team_name, history, metric):
                if team_name not in history or not history[team_name]:
                    return 0
                
                last_5 = history[team_name][-5:]
                decay_factor = 0.85
                weighted_val = 0
                total_weight = 0
                
                for i, stats in enumerate(reversed(last_5)):
                    weight = decay_factor ** i
                    val = stats.get(metric, 0)
                    weighted_val += val * weight
                    total_weight += weight
                    
                return weighted_val / total_weight if total_weight > 0 else 0

            
            home_form = get_form(home_team, team_history)
            away_form = get_form(away_team, team_history)
            
    # Calculate Event Form (Rolling Average)
            metrics = [
                'pass_accuracy', 'tackles_won', 'duels_won', 'shots_on_target', 'possession_lost', 'shot_accuracy', 'conversion_rate',
                # Tier 1 Metrics
                'expected_assists', 'big_chance_missed', 'tackle_success', 'interceptions', 'dribble_success'
            ]
            home_event_feats = {m: get_event_form(home_team, team_event_history, m) for m in metrics}
            away_event_feats = {m: get_event_form(away_team, team_event_history, m) for m in metrics}
            
            # Update History (Points)
            if home_score > away_score: h_pts, a_pts = 3, 0
            elif home_score == away_score: h_pts, a_pts = 1, 1
            else: h_pts, a_pts = 0, 3
                
            if home_team not in team_history: team_history[home_team] = []
            if away_team not in team_history: team_history[away_team] = []
            team_history[home_team].append(h_pts)
            team_history[away_team].append(a_pts)
            
            # Update Event History
            # Calculate team stats from match_rows
            home_rows = match_rows[match_rows['is_home'] == 1]
            away_rows = match_rows[match_rows['is_home'] == 0]
            
            def get_team_stats(rows):
                total_passes = rows['total_passes'].sum()
                accurate_passes = rows['passes_completed'].sum()
                pass_acc = accurate_passes / total_passes if total_passes > 0 else 0
                
                total_shots = rows['total_shots'].sum()
                shots_on_target = rows['shots_on_target'].sum()
                goals = rows['goals'].sum()
                
                shot_acc = shots_on_target / total_shots if total_shots > 0 else 0
                conversion = goals / shots_on_target if shots_on_target > 0 else 0
                
                # Tier 1 Stats
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
                    # Tier 1
                    'expected_assists': rows['expected_assists'].sum(),
                    'big_chance_missed': rows['big_chance_missed'].sum(),
                    'tackle_success': tackle_success,
                    'interceptions': rows['interceptions'].sum(),
                    'dribble_success': dribble_success
                }

            h_stats = get_team_stats(home_rows)
            a_stats = get_team_stats(away_rows)
            
            if home_team not in team_event_history: team_event_history[home_team] = []
            if away_team not in team_event_history: team_event_history[away_team] = []
            
            team_event_history[home_team].append(h_stats)
            team_event_history[away_team].append(a_stats)
            
            # Store Training Example
            example = {
                'match_id': mid, 'date': match_date, 'league_id': match_rows.iloc[0]['league_id'],
                'home_team': home_team, 'away_team': away_team,
                'home_score': home_score, 'away_score': away_score,
                'Δ_start': delta_start, 'Δ_sub': delta_sub,
                # Event Features (Deltas of Forms)
                'Δ_pass_accuracy': home_event_feats['pass_accuracy'] - away_event_feats['pass_accuracy'],
                'Δ_tackles': home_event_feats['tackles_won'] - away_event_feats['tackles_won'],
                'Δ_duels': home_event_feats['duels_won'] - away_event_feats['duels_won'],
                'Δ_shots_on_target': home_event_feats['shots_on_target'] - away_event_feats['shots_on_target'],
                'Δ_possession_lost': home_event_feats['possession_lost'] - away_event_feats['possession_lost'],
                'Δ_shot_accuracy': home_event_feats.get('shot_accuracy', 0) - away_event_feats.get('shot_accuracy', 0),
                'Δ_conversion_rate': home_event_feats.get('conversion_rate', 0) - away_event_feats.get('conversion_rate', 0),
                # Tier 1 Features
                'Δ_expected_assists': home_event_feats.get('expected_assists', 0) - away_event_feats.get('expected_assists', 0),
                'Δ_big_chance_missed': home_event_feats.get('big_chance_missed', 0) - away_event_feats.get('big_chance_missed', 0),
                'Δ_tackle_success': home_event_feats.get('tackle_success', 0) - away_event_feats.get('tackle_success', 0),
                'Δ_interceptions': home_event_feats.get('interceptions', 0) - away_event_feats.get('interceptions', 0),
                'Δ_dribble_success': home_event_feats.get('dribble_success', 0) - away_event_feats.get('dribble_success', 0)
            }
            training_data.append(example)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing match {mid}: {e}")
            # import traceback
            # traceback.print_exc()
            error_count += 1
            continue

    print(f"✅ Processed {success_count} matches. Failed: {error_count}")
    
    if not training_data:
        print("❌ No training data generated.")
        return

    # 4. Train Models
    models_dir = "src/models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # --- A. Standard Model (Baseline) ---
    print("\n🧠 Training Standard Skellam Model (Baseline)...")
    train_df = pd.DataFrame(training_data)
    print(f"   Initial training examples: {len(train_df)}")
    
    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Δ_start', 'Δ_sub', 'home_score', 'away_score'])
    print(f"   After cleaning (NaN/Inf removal): {len(train_df)}")
    
    cleaned_training_data = train_df.to_dict('records')
    
    # Split Train/Test
    train_size = int(len(cleaned_training_data) * 0.8)
    train_set = cleaned_training_data[:train_size]
    test_set = cleaned_training_data[train_size:]
    
    print(f"   Training Set: {len(train_set)} matches")
    
    try:
        # Standard Features implicitly used by default if feature_columns is None
        predictor.fit_skellam_model(train_set)
        predictor.save_model(os.path.join(models_dir, "skellam_model_avg.pkl"))
        
        # Evaluate Standard Model
        print("📉 Evaluating Standard Model...")
        correct_std = 0
        total_std = 0
        for match in test_set:
            pred = predictor.predict_match_outcome(match)
            
            if match['home_score'] > match['away_score']: actual = 'home_win'
            elif match['home_score'] == match['away_score']: actual = 'draw'
            else: actual = 'away_win'
            
            probs = {'home_win': pred['prob_home_win'], 'draw': pred['prob_draw'], 'away_win': pred['prob_away_win']}
            predicted = max(probs, key=probs.get)
            if predicted == actual: correct_std += 1
            total_std += 1
        
        acc_std = correct_std / total_std if total_std > 0 else 0
        print(f"🏆 Standard Model Test Accuracy: {acc_std:.2%}")

    except Exception as e:
        print(f"❌ Standard Model training failed: {e}")

    # --- B. Tier 1 Model (Advanced) ---
    print("\n💎 Training Tier 1 Skellam Model (Advanced)...")
    TIER1_LEAGUES = [7, 17, 8, 23, 35, 34, 679]
    TIER1_FEATURES = [
        'Δ_start', 'Δ_sub',
        'Δ_pass_accuracy', 'Δ_tackles', 'Δ_duels', 'Δ_shots_on_target', 'Δ_possession_lost', 'Δ_shot_accuracy', 'Δ_conversion_rate',
        'Δ_expected_assists', 'Δ_big_chance_missed', 'Δ_tackle_success', 'Δ_interceptions', 'Δ_dribble_success'
    ]
    
    # Fix league_id type mismatch (convert to int)
    print(f"   Before league filter: {len(train_df)} matches")
    print(f"   Unique league_ids in data (before conversion): {train_df['league_id'].unique()}")
    
    # Convert league_id to int, handling strings and already-int values
    train_df['league_id'] = pd.to_numeric(train_df['league_id'], errors='coerce').fillna(0).astype(int)
    print(f"   Unique league_ids in data (after conversion): {train_df['league_id'].unique()}")
    
    # Filter for Tier 1 Leagues
    tier1_df = train_df[train_df['league_id'].isin(TIER1_LEAGUES)].copy()
    print(f"   After league filter: {len(tier1_df)} matches")
    
    if len(tier1_df) > 100:
        tier1_data = tier1_df.to_dict('records')
        t1_train_size = int(len(tier1_data) * 0.8)
        t1_train_set = tier1_data[:t1_train_size]
        t1_test_set = tier1_data[t1_train_size:]
        
        print(f"   Tier 1 Training Set: {len(t1_train_set)} matches")
        print(f"   Tier 1 Test Set: {len(t1_test_set)} matches")
        
        try:
            predictor.fit_skellam_model(t1_train_set, feature_columns=TIER1_FEATURES)
            predictor.save_model(os.path.join(models_dir, "skellam_model_tier1.pkl"))
            
            # Evaluate Tier 1
            print("📉 Evaluating Tier 1 Model...")
            correct = 0
            total = 0
            for match in t1_test_set:
                pred = predictor.predict_match_outcome(match) # Will use Tier 1 features because model is loaded
                
                if match['home_score'] > match['away_score']: actual = 'home_win'
                elif match['home_score'] == match['away_score']: actual = 'draw'
                else: actual = 'away_win'
                
                probs = {'home_win': pred['prob_home_win'], 'draw': pred['prob_draw'], 'away_win': pred['prob_away_win']}
                predicted = max(probs, key=probs.get)
                if predicted == actual: correct += 1
                total += 1
            
            acc = correct / total if total > 0 else 0
            print(f"🏆 Tier 1 Test Accuracy: {acc:.2%}")
            
        except Exception as e:
            print(f"❌ Tier 1 Model training failed: {e}")
    else:
        print("⚠️ Not enough data for Tier 1 Model.")

    # --- C. XGBoost Model (Experimental) ---
    print("\n🚀 Training XGBoost Model (Experimental)...")
    try:
        xgb_predictor = XGBoostPredictor()
        # Use Standard Features for now + Tier 1 features if available?
        # Let's use the full set of Tier 1 features for XGBoost as it handles noise well
        XGB_FEATURES = TIER1_FEATURES
        
        # Use the same Tier 1 dataset (or full dataset if we want)
        # Let's use the full training set but with Tier 1 features (imputing 0 for missing)
        # Actually, let's stick to the Tier 1 subset to be fair, or full set if we handle missing cols
        
        # For simplicity, let's train on the Tier 1 dataset as it has the advanced metrics
        if len(tier1_df) > 100:
            print(f"   Training XGBoost on {len(t1_train_set)} matches (Tier 1 Subset)...")
            xgb_predictor.fit(t1_train_set, feature_columns=XGB_FEATURES)
            xgb_predictor.save_model(os.path.join(models_dir, "xgboost_model.pkl"))
            
            # Evaluate XGBoost
            print("📉 Evaluating XGBoost Model...")
            correct_xgb = 0
            total_xgb = 0
            for match in t1_test_set:
                pred = xgb_predictor.predict_match_outcome(match)
                
                if match['home_score'] > match['away_score']: actual = 'home_win'
                elif match['home_score'] == match['away_score']: actual = 'draw'
                else: actual = 'away_win'
                
                probs = {'home_win': pred['prob_home_win'], 'draw': pred['prob_draw'], 'away_win': pred['prob_away_win']}
                predicted = max(probs, key=probs.get)
                if predicted == actual: correct_xgb += 1
                total_xgb += 1
            
            acc_xgb = correct_xgb / total_xgb if total_xgb > 0 else 0
            print(f"🏆 XGBoost Test Accuracy: {acc_xgb:.2%}")
        else:
            print("⚠️ Not enough Tier 1 data for XGBoost.")

    except Exception as e:
        print(f"❌ XGBoost training failed: {e}")
    
    # 5. Evaluate
    print("📉 Evaluating on Test Set...")
    correct_outcomes = 0
    total_test = 0
    
    for match in test_set:
        pred = predictor.predict_match_outcome(match)
        
        # Determine actual outcome
        if match['home_score'] > match['away_score']:
            actual = 'home_win'
        elif match['home_score'] == match['away_score']:
            actual = 'draw'
        else:
            actual = 'away_win'
            
        # Determine predicted outcome (highest prob)
        probs = {
            'home_win': pred['prob_home_win'],
            'draw': pred['prob_draw'],
            'away_win': pred['prob_away_win']
        }
        predicted = max(probs, key=probs.get)
        
        if predicted == actual:
            correct_outcomes += 1
        total_test += 1
        
    accuracy = correct_outcomes / total_test if total_test > 0 else 0
    print(f"🏆 Test Set Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    train_pipeline()
