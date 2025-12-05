import os
import json
import time
from match_collector import LeagueIds, SeasonCollector, MatchCollector

# Directory for saving data
OUTPUT_DIR = "json_data/new_league_info"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_json(data, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"💾 Saved data to {filepath}")

def main():
    print("🚀 Starting Data Collection...")
    
    # 1. Get League IDs
    # Note: You can manually define LEAGUE_IDS if you want specific ones, 
    # or use LeagueIds class if it works reliably.
    # For now, let's use the explicit list from your system or fetch them.
    
    # Using the LeagueIds class from match_collector.py
    league_ids_collector = LeagueIds()
    league_ids_collector.getLeagueIds()
    league_ids = league_ids_collector.league_ids
    
    print(f"📋 Found {len(league_ids)} leagues to process: {league_ids}")
    
    for league_id in league_ids:
        # Check if league data already exists
        matches_filename = f"league_{league_id}.json"
        filepath = os.path.join(OUTPUT_DIR, matches_filename)
        
        if os.path.exists(filepath):
            print(f"\n⏭️  Skipping League ID: {league_id} (Data already exists)")
            continue

        print(f"\n🏆 Processing League ID: {league_id}")
        
        try:
            # 2. Collect Seasons & Standings
            season_collector = SeasonCollector(league_id)
            season_collector.getLeagueSeasons() # This populates season_events and season_standing
            
            # Save Standings
            standings_filename = f"standings_{league_id}.json"
            save_json(season_collector.season_standing, standings_filename)
            
            # 3. Collect Match Details (Odds, Lineups)
            # season_collector.season_events contains basic match info. 
            # We need to enrich it using MatchCollector.
            
            enriched_matches = []
            total_matches = len(season_collector.season_events)
            print(f"   ⚽ Found {total_matches} matches. Fetching details...")
            
            consecutive_missing_lineups = 0
            for idx, match_event in enumerate(season_collector.season_events):
                print(f"      Processing match {idx+1}/{total_matches} (ID: {match_event.get('id')})...")
                
                # Create a match object compatible with MatchCollector
                # MatchCollector expects a dict with 'match_id' and 'league'
                # The event object from Sofascore usually has 'id'
                
                match_input = {
                    "match_id": match_event.get("id"),
                    "league": league_id,
                    "home_team":match_event.get("homeTeam").get("name"),
                    "away_team":match_event.get("awayTeam").get("name"),
                    "home_team_id":match_event.get("homeTeam").get("id"),
                    "away_team_id":match_event.get("awayTeam").get("id"),
                    "match_date":match_event.get("startTimestamp"),
                    "home_score":match_event.get("homeScore").get("normaltime"),
                    "away_score":match_event.get("awayScore").get("normaltime"),
                    "season":match_event.get("season").get("id")
                }
                # Initialize MatchCollector
                match_collector = MatchCollector(match_input)
                
                # Fetch Lineup & Odds
                match_collector.getLineUp() # This populates self.complete_match_obj
                if match_collector.complete_match_obj:
                    enriched_matches.append(match_collector.complete_match_obj)
                    consecutive_missing_lineups = 0
                else:
                    print(f"      ⚠️ No complete data for match {match_event.get('id')}")
                    consecutive_missing_lineups += 1
                
                if consecutive_missing_lineups >= 4:
                    print(f"      🛑 Stopping league {league_id} due to 4 consecutive missing lineups.")
                    break
                
                # Be nice to the API
                time.sleep(2) 
            
            # Save Enriched Matches
            matches_filename = f"league_{league_id}.json"
            obj={
                "count":len(enriched_matches),
                "data":enriched_matches 
            }
            save_json(obj, matches_filename)
            
        except Exception as e:
            print(f"❌ Error processing league {league_id}: {e}")
            continue

    print("\n✅ Data Collection Complete!")

if __name__ == "__main__":
    main()
