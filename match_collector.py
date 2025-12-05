import time
import numpy as np
from datetime import datetime
from typing import Any
from defaults import DefaultMixin,calculate_odds
from cdriver import chromeDriver
from mapper import AutomaticPositionMapper
LEAGUE_NAMES=[
    "UEFA Champions League",
    "UEFA Europa League",
    "Premier League",
    "LaLiga",
    "Bundesliga",
    "Serie A",
    "Ligue 1"
]

                    
class LeagueIds(DefaultMixin):
    def __init__(self):
        self.url="https://www.sofascore.com/api/v1/config/default-unique-tournaments/KE/football"
        self.league_ids=[]
        self.leagues_found=False
        self.driver=chromeDriver(headless=True)
    def getLeagueIds(self):
        response:dict=self.getUrlResponse(self.driver,self.url)
        if "uniqueTournaments" in response.keys():
            self.leagues_found=True
            leagues:list[dict]=response.get("uniqueTournaments")
            for league in leagues:
                name=league.get("name")
                if name in LEAGUE_NAMES:
                    league_id=league.get("id")
                    self.league_ids.append(league_id)
        else:
            print(response)
        self.driver.close()        


# collects events for a single season
class EventsCollector(DefaultMixin):
    def __init__(self,league_id:int,season_id:int):
        self.league_id=league_id
        self.season_id=season_id
        self.driver=chromeDriver(headless=True)
        self.events:list=[]
    def getMatches(self):
        for i in range(6):
            url=f"https://www.sofascore.com/api/v1/unique-tournament/{self.league_id}/season/{self.season_id}/events/last/{i}"
            response:dict=self.getUrlResponse(self.driver,url)
            time.sleep(3)
            if "events" in response.keys():
                events=response.get("events")
                self.events.extend(events)
                print(f"Page {i} completed sucessfully")
            elif "error" in response.keys():
                print(f"{response}. No mo events at {i}")
                break
            
        self.driver.close()
            
            
# collects 
class SeasonCollector(DefaultMixin):
    def __init__(self,league_id:int):
        self.league_id=league_id
        self.url=f"https://api.sofascore.com/api/v1/unique-tournament/{self.league_id}/seasons"
        self.driver=chromeDriver(headless=True)
        self.season_events=[]
        self.season_standing={}
    def getLeagueSeasons(self):
        response:dict=self.getUrlResponse(self.driver,self.url)
        if "seasons" in response.keys():
            seasons:list[dict[str,Any]]=response.get("seasons")
            current_year=datetime.today().year
            for season in seasons:
                year:str=season.get("year")
                start_year,_=[int("20"+i) for i in year.split("/")]
                if current_year - start_year <=2:
                    season_id=season.get("id")
                    event_collector=EventsCollector(self.league_id,season_id)
                    event_collector.getMatches()
                    self.season_standing[season_id]=self.getSeasonStanding(season_id)
                    self.season_events.extend(event_collector.events)
                    time.sleep(3)
                    print(f"Competed collecting data for {season_id} season")
            
            self.driver.close()
        else:
            print(response)
            
    def getSeasonStanding(self,season_id):
        url=f"https://www.sofascore.com/api/v1/unique-tournament/{self.league_id}/season/{season_id}/standings/total"
        response=self.getUrlResponse(self.driver,url)
        season_standing=[]
        if "standings" in response.keys():
            rows:list[dict]=response["standings"][0]["rows"]
            for team_data in rows:
                team_dict = {
                    "team": team_data.get("team").get("name"),
                    "team_id": team_data.get("id"),
                    "position": team_data.get("position"),
                    "matches_played": team_data.get("matches"),
                    "wins": team_data.get("wins"),
                    "goals_scored": team_data.get("scoresFor", 0),
                    "goals_conceded": team_data.get("scoresAgainst", 0),
                    "score_difference": team_data.get("scoresFor", 0) - team_data.get("scoresAgainst", 0),
                    "losses": team_data.get("losses", 0),
                    "draws": team_data.get("draws", 0),
                    "points": team_data.get("points")
                }
                season_standing.append(team_dict)
        return season_standing
            
        
class MatchCollector(DefaultMixin):
    def __init__(self,match_obj:dict):
        self.match_obj=match_obj
        self.match_id=match_obj.get("match_id")
        self.league=match_obj.get("league")
        self.driver=chromeDriver(headless=True)
        self.complete_match_obj={}
        self.phi = 0.0062
    
    def getLineUp(self) -> dict|None:
        url=f"https://www.sofascore.com/api/v1/event/{self.match_id}/lineups"  
        try:
                   
            response=self.getUrlResponse(self.driver,url)
            if "confirmed" in response.keys() and response["confirmed"]:
                print("Lineup Found")
                home_players=response["home"]["players"]
                away_players=response["away"]["players"]
                home_formation=response["home"]["formation"]
                away_formation=response["away"]["formation"]
                
                
                player_mapper=AutomaticPositionMapper()
                hm_players,h_formation=player_mapper.assign_positions(home_players,home_formation)
                aw_players,a_formation=player_mapper.assign_positions(away_players,away_formation)
                
                
                h_players=self.getPlayersData(hm_players,True)
                a_players=self.getPlayersData(aw_players,False)
                odds=self.getOdds()
                obj={
                    "home_players":h_players,
                    "away_players":a_players,
                    "home_formation":h_formation,
                    "away_formation":a_formation
                }
                
                self.complete_match_obj = self.match_obj|obj|odds
            
            else:
                print(f"Lineup Error")
                if "error" in response.keys():
                    print(response)
                else:
                    print(f"Match was not confirmed for {self.match_id}")
        except Exception as e:
            print(e)
        
        finally:
            self.driver.close()
            time.sleep(2)
    def getOdds(self):
        url=f"https://www.sofascore.com/api/v1/event/{self.match_id}/odds/1/all"
        response=self.getUrlResponse(self.driver,url)
        odd_obj={}
        if "markets" in response.keys():
            for odds in response["markets"]:
                odd_type:str|None=odds.get("marketGroup")
                market_name=odds.get("marketName")
                if odd_type and odd_type == "1X2" and market_name == "Full time":
                    choices:list[dict]|None=odds.get("choices")
                    if choices:
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
        return odd_obj
    
                        
                    
    def getPlayersData(self,players:list[dict],is_home:bool) -> list[dict]:
       
        players_data=[]
        for player in players:
            player_obj:dict=player.get("player")
            player_id=player_obj.get("id")
            player_name=player_obj.get("name")
            player_rating=player.get("statistics").get("rating") if player.get("statistics") else 0
            substitute=player.get("substitute")
            minutes_played=player.get("statistics").get("minutesPlayed",0) if player.get("statistics") else 0
            c_player_obj={
                "player_id":player_id,
                "player_name":player_name,
                "rating":player_rating,
                "is_home":1 if is_home else 0,
                "substitute":substitute,
                "minutes_played":minutes_played,
                "position":player.get("standard_position"),
                "specific_position":player.get("specific_position"),
                "statistics":player.get("statistics")
            }
            players_data.append(c_player_obj)
            
        return players_data
    def compute_weight(self,days_diff):
        return np.exp(-self.phi * (days_diff / 3.5))
            
    