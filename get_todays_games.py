import json
import re
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
import time


from selenium.webdriver import Chrome,ChromeOptions
from defaults import calculate_odds

date_today_str=datetime.today().date()
date_today_str=date_today_str.strftime("%Y-%m-%d")



def chromeDriver(headless:bool) -> Chrome:
    options=ChromeOptions()
    options.add_argument("--no-sandbox")
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    driver=Chrome(options=options)
    return driver


def getOdds(match_id,driver:Chrome):
        url=f"https://www.sofascore.com/api/v1/event/{match_id}/odds/1/all"
        driver.get(url)
        results=driver.page_source
        soup = BeautifulSoup(results, "html.parser")
        response=json.loads(soup.find("pre").text)
        odd_obj={}
        if "markets" in response.keys():
            for odds in response["markets"]:
                odd_type:str|None=odds.get("marketGroup")
                market_name=odds.get("marketName")
                if odd_type and odd_type == "1X2" and market_name == "Full time":
                    choices:list[dict]|None=odds.get("choices")
                    home_odds=choices[0].get("fractionalValue")
                    draw_odds=choices[1].get("fractionalValue")
                    away_odds=choices[2].get("fractionalValue")
                    
                    odd_obj["home_odds"]=round(calculate_odds(home_odds),2)
                    odd_obj["draw_odds"]=round(calculate_odds(draw_odds),2)
                    odd_obj["away_odds"]=round(calculate_odds(away_odds),2)
        
        if odd_obj:
            print("Odds found")
        else:
            print("Odds not found")
        time.sleep(2)
        return odd_obj

url=f"https://www.sofascore.com/api/v1/sport/football/scheduled-events/{date_today_str}"
driver=chromeDriver(headless=True)
driver.get(url)
results=driver.page_source
soup = BeautifulSoup(results, "html.parser")
games_today=json.loads(soup.find("pre").text)
driver.close()

event_data = []

leagues = [
    {"league": "Premier League", "country": "England"},
    {"league": "Bundesliga", "country": "Germany"},
    {"league": "Laliga", "country": "Spain"},
    {"league": "LaLiga", "country": "Spain"},
    {"league": "Series A", "country": "Italy"},
    {"league": "UEFA Champions League", "country": "Europe"},
    {"league": "Ligue 1", "country": "France"},
    {"league": "UEFA Europa League", "country": "Europe"},
]

# Make a lookup set for quick matching
league_lookup = {(l["league"], l["country"]) for l in leagues}

u_games = re.compile(r"U[0-9]{2}", re.IGNORECASE)
women = re.compile(r"women", re.IGNORECASE)
driver=chromeDriver(headless=True)
for i in games_today["events"]:
    tournament = i["tournament"]["name"]
    country = i["tournament"]["category"]["name"]

    home_team = i['homeTeam']["name"]
    away_team = i['awayTeam']["name"]
    status= i["status"]["type"]
    # Skip women + youth teams
    if women.search(tournament) or u_games.search(home_team) or u_games.search(away_team):
        continue
    # Filter only tournaments matching league + country
    if (tournament, country) in league_lookup and status=="notstarted":
        obj={
            "name": tournament,
            "country": country,
            "home team": home_team,
            "away team": away_team,
            "slug": i["tournament"]["slug"],
            "home_id": i['homeTeam']["id"],
            "away_id": i['awayTeam']["id"],
            "start_time": i["startTimestamp"],
            "match_id": i["id"],
            "tournament_id": i["tournament"]["uniqueTournament"]["id"],
            "season":i["season"]["id"]
        }
        odds=getOdds(i["id"],driver)
        obj|=odds
        print(home_team, away_team)
        event_data.append(obj)

driver.close()


df = pd.DataFrame(event_data)

# ensure datetime type (timezone aware)
df["start_dt"] = pd.to_datetime(df["start_time"], unit="s", utc=True).dt.tz_convert("Africa/Nairobi")


today = pd.Timestamp.now().date()
df_today = df[df["start_dt"].dt.date == today].copy()   # <-- make a copy to avoid SettingWithCopyWarning

# Create HH:MM column for grouping
df_today["match_time"] = df_today["start_dt"].dt.strftime("%H:%M")

# convert to string for JSON
df_today["start_dt"] = df_today["start_dt"].astype(str)

# sort chronologically
df_today = df_today.sort_values("start_dt").reset_index(drop=True)

# group by match_time
grouped = {
    time_str: group.drop(columns=["match_time"]).to_dict(orient="records")
    for time_str, group in df_today.groupby("match_time", sort=False) # sort=False preserves the sort order from above
}


with open(f"json_data/games_today/{date_today_str}-games.json","w") as file:
    json.dump(grouped,file)