import pandas as pd
import numpy as np
import sys
import os

# Mock functions from train_model.py
def get_event_form(team_name, history_dict, metric):
    if team_name not in history_dict:
        return 0.0
    
    recent = history_dict[team_name][-5:]
    if not recent: return 0.0
    
    # Extract metric values
    values = [g[metric] for g in recent]
    weights = np.exp(np.linspace(-0.5, 0, len(values))) # Less decay for stats
    return np.average(values, weights=weights)

def main():
    print("📂 Loading Master Data...")
    try:
        master_df = pd.read_csv("src/data/master_player_stats.csv")
        master_df['date'] = pd.to_datetime(master_df['date'])
        master_df = master_df.sort_values('date')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Loaded {len(master_df)} rows.")

    # Group by match
    match_groups = master_df.groupby('match_id')
    
    training_data = []
    team_event_history = {}
    
    TIER1_FEATURES = [
        'Δ_expected_assists', 'Δ_big_chance_missed', 'Δ_tackle_success', 'Δ_interceptions', 'Δ_dribble_success'
    ]
    
    print("🔄 Processing matches...")
    for mid, match_rows in match_groups:
        home_rows = match_rows[match_rows['is_home'] == 1]
        away_rows = match_rows[match_rows['is_home'] == 0]
        
        if home_rows.empty or away_rows.empty:
            continue
            
        home_team = home_rows.iloc[0]['team']
        away_team = away_rows.iloc[0]['team']
        
        # Calculate Event Form (Rolling Average)
        metrics = [
            'pass_accuracy', 'tackles_won', 'duels_won', 'shots_on_target', 'possession_lost', 'shot_accuracy', 'conversion_rate',
            # Tier 1 Metrics
            'expected_assists', 'big_chance_missed', 'tackle_success', 'interceptions', 'dribble_success'
        ]
        home_event_feats = {m: get_event_form(home_team, team_event_history, m) for m in metrics}
        away_event_feats = {m: get_event_form(away_team, team_event_history, m) for m in metrics}
        
        # Calculate team stats from match_rows
        def get_team_stats(rows):
            total_passes = rows['total_passes'].sum()
            accurate_passes = rows['accurate_passes'].sum()
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
            'match_id': mid,
            'league_id': match_rows.iloc[0]['league_id'],
            # Tier 1 Features
            'Δ_expected_assists': home_event_feats.get('expected_assists', 0) - away_event_feats.get('expected_assists', 0),
            'Δ_big_chance_missed': home_event_feats.get('big_chance_missed', 0) - away_event_feats.get('big_chance_missed', 0),
            'Δ_tackle_success': home_event_feats.get('tackle_success', 0) - away_event_feats.get('tackle_success', 0),
            'Δ_interceptions': home_event_feats.get('interceptions', 0) - away_event_feats.get('interceptions', 0),
            'Δ_dribble_success': home_event_feats.get('dribble_success', 0) - away_event_feats.get('dribble_success', 0)
        }
        training_data.append(example)

    df = pd.DataFrame(training_data)
    print(f"Generated {len(df)} training examples.")
    
    TIER1_LEAGUES = [7, 17, 8, 23, 35, 34, 679]
    # Handle string/int mismatch by converting to int if possible or checking both
    # The CSV output showed integers, so let's try to ensure type match
    
    # Filter
    t1_df = df[df['league_id'].isin(TIER1_LEAGUES)]
    print(f"Filtered Tier 1 examples: {len(t1_df)}")
    
    print("\nChecking for NaNs in Tier 1 Features:")
    print(t1_df[TIER1_FEATURES].isnull().sum())
    
    print("\nChecking for Infs in Tier 1 Features:")
    print(np.isinf(t1_df[TIER1_FEATURES]).sum())
    
    # Check for Zeros (just in case)
    print("\nChecking for Zeros in Tier 1 Features:")
    print((t1_df[TIER1_FEATURES] == 0).sum())

if __name__ == "__main__":
    main()
