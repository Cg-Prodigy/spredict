import json
import os
import time
from datetime import datetime

from match_collector import SeasonCollector, EventsCollector, chromeDriver


leagues=[7, 17, 8, 23, 35, 34, 679]
class WeeklyUpdater:
    def __init__(self):
        self.json_dir = "json_data/league_info"
        
    def update_leagues(self):
        print(f"🚀 Starting Weekly Update for {len(leagues)} leagues...")
        
        for league_id in leagues:
            print(f"\n🔄 Processing League ID: {league_id}")
            
            # 1. Get Seasons
            season_collector = SeasonCollector(league_id)
            response = season_collector.getUrlResponse(season_collector.driver, season_collector.url)
            season_collector.driver.close()
            
            if "seasons" not in response:
                print(f"❌ Could not fetch seasons for league {league_id}")
                continue
                
            seasons = response.get("seasons")
            current_season_id = None
            
            # Find current season (25/26)
            # Logic: Look for year "25/26" or similar. 
            # Or just take the first one as it's usually the latest?
            # User said: "current season is 25/26"
            
            for season in seasons:
                if season.get("year") == "25/26":
                    current_season_id = season.get("id")
                    break
            
            if not current_season_id:
                # Fallback: Try to find the most recent one by ID or year parsing if 25/26 string match fails
                # But user was specific. Let's try to be robust.
                # If 25/26 not found, maybe it's 2025 or 2026?
                # Let's just take the first one (index 0) as it is typically the current one in Sofascore API
                print(f"⚠️ '25/26' season not found explicitly. Using the most recent season from list.")
                current_season_id = seasons[0].get("id")
                
            print(f"📅 Current Season ID: {current_season_id}")
            
            # 2. Fetch Latest Matches (Page 0 only)
            print("📡 Fetching latest matches (Page 0)...")
            # Custom EventsCollector logic for Page 0
            driver = chromeDriver(headless=True)
            url = f"https://www.sofascore.com/api/v1/unique-tournament/{league_id}/season/{current_season_id}/events/last/0"
            
            # We need to use the mixin method, but we can't easily instantiate EventsCollector just for one call 
            # without modifying it or subclassing. 
            # Let's just use the driver directly here since we have it.
            # Wait, we need the getUrlResponse method from DefaultMixin.
            # Let's instantiate EventsCollector but override getMatches or just call getUrlResponse manually.
            
            # Better: Use EventsCollector but only for page 0
            # User said: "instead of collecting matches to range(6), i will be 0"
            
            # Let's create a temporary instance to use its methods
            temp_collector = EventsCollector(league_id, current_season_id)
            # We can't use getMatches() because it loops range(6).
            # We'll just do the request manually using the helper
            response = temp_collector.getUrlResponse(driver, url)
            driver.close()
            
            new_events = []
            if "events" in response:
                new_events = response.get("events")
                print(f"✅ Found {len(new_events)} matches on Page 0")
            else:
                print(f"❌ No events found on Page 0: {response}")
                continue
                
            # 3. Load Existing JSON
            json_path = os.path.join(self.json_dir, f"league_{league_id}.json")
            existing_data = []
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    try:
                        loaded_content = json.load(f)
                        if isinstance(loaded_content, dict) and "data" in loaded_content:
                            existing_data = loaded_content["data"]
                        elif isinstance(loaded_content, list):
                            existing_data = loaded_content
                        else:
                            print("⚠️ Unexpected JSON structure. Starting fresh.")
                            existing_data = []
                    except json.JSONDecodeError:
                        print("⚠️ Error decoding existing JSON. Starting fresh.")
                        existing_data = []
            else:
                print("⚠️ JSON file not found. Creating new.")
                
            # 4. Compare and Merge
            # Existing data uses 'match_id', new API data uses 'id'
            existing_ids = {event.get('match_id') for event in existing_data}
            added_count = 0
            
            # We need to fetch DETAILS for new matches? 
            # The 'events' endpoint usually returns basic info. 
            # Do we need full details (lineups, etc.)?
            # The user said: "we add them to the leageu_info jsons"
            # In match_collector.py, MatchCollector fetches lineups.
            # If we just add the event object from 'last/0', it might lack player stats.
            # BUT, looking at `match_collector.py`, `EventsCollector` just collects the event objects.
            # It seems `MatchCollector` is used later or separately?
            # Wait, `league_{id}.json` usually contains FULL match data (players, stats).
            # If we just append the summary from `events/last/0`, we miss the stats needed for the model.
            
            # CRITICAL: We need to fetch full match details for the new events.
            # The user didn't explicitly say "fetch details", but implied it by saying "add them to league_info".
            # League info files are used for training, so they MUST have player stats.
            
            from match_collector import MatchCollector
            
            for event in new_events:
                match_id = event['id']
                if match_id not in existing_ids:
                    print(f"   ➕ New Match Found: {event['homeTeam']['name']} vs {event['awayTeam']['name']} ({match_id})")
                    
                    # Fetch full details
                    # MatchCollector expects a dict with match_id and league
                    match_obj_input = {
                        "match_id": match_id,
                        "league": event['tournament']['name']
                    }
                    
                    collector = MatchCollector(match_obj_input)
                    collector.getLineUp() # This fetches players and stats
                    
                    if collector.complete_match_obj:
                        # Add to existing data
                        existing_data.append(collector.complete_match_obj)
                        added_count += 1
                        # Add to ID set to prevent duplicates in this run
                        existing_ids.add(match_id)
                    else:
                        print(f"   ⚠️ Could not fetch details for match {match_id}")
                    
                    # Sleep to be polite to API
                    time.sleep(2)
            
            if added_count > 0:
                print(f"💾 Saving {added_count} new matches to {json_path}")
                
                # Save in the same format: {"count": N, "data": [...]}
                output_data = {
                    "count": len(existing_data),
                    "data": existing_data
                }
                
                with open(json_path, 'w') as f:
                    json.dump(output_data, f, indent=4)
            else:
                print("✨ No new matches to add.")
                
        print("\n✅ Weekly Update Complete!")

if __name__ == "__main__":
    updater = WeeklyUpdater()
    updater.update_leagues()
