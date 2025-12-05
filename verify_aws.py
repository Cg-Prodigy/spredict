import pandas as pd
from aws_calculator import AWSRatingCalculator
from datetime import datetime

def verify_aws():
    print("🚀 Starting AWS Calculator Verification...")
    
    # 1. Load Data
    try:
        df = pd.read_csv("data/master_player_stats.csv")
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Loaded {len(df)} records from master_player_stats.csv")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # 2. Setup Calculator
    calculator = AWSRatingCalculator()
    
    # 3. Pick a test case
    # Let's pick a match from the middle of the dataset to ensure some history exists
    # Sort by date to find a good candidate
    df = df.sort_values('date')
    
    # Pick a date roughly in the middle
    mid_idx = len(df) // 2
    test_match_date = df.iloc[mid_idx]['date']
    test_match_id = df.iloc[mid_idx]['match_id']
    
    # Get players from that match
    match_players = df[df['match_id'] == test_match_id]['player_id'].unique().tolist()
    
    print(f"\n🧪 Test Case:")
    print(f"   Match Date: {test_match_date}")
    print(f"   Match ID: {test_match_id}")
    print(f"   Players to rate: {len(match_players)}")
    
    # 4. Run Calculation
    start_time = datetime.now()
    ratings = calculator.calculate_ratings_for_match_day(test_match_date, match_players, df)
    end_time = datetime.now()
    
    # 5. Results
    print(f"\n⏱️ Calculation took: {(end_time - start_time).total_seconds():.2f} seconds")
    print("\n📊 Sample Ratings:")
    for pid in match_players[:5]: # Show first 5
        p_name = df[df['player_id'] == pid]['player_name'].iloc[0]
        rating = ratings.get(pid, "N/A")
        print(f"   - {p_name} (ID: {pid}): {rating:.2f}")
        
    # 6. Sanity Checks
    if not ratings:
        print("\n❌ Error: No ratings returned.")
    else:
        vals = list(ratings.values())
        avg_r = sum(vals) / len(vals)
        print(f"\n✅ Success! Average Rating: {avg_r:.2f}")
        if avg_r < 1 or avg_r > 10:
             print("⚠️ Warning: Average rating is outside expected 1-10 range.")

if __name__ == "__main__":
    verify_aws()
