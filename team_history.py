
import time
import numpy as np
from datetime import datetime


from mapper import AutomaticPositionMapper
from defaults import DefaultMixin
from cdriver import chromeDriver


class TeamData(DefaultMixin):
    def __init__(self, team_id:int,phi: float = 0.0062, psi: float = 2.00):
        self.driver=chromeDriver(headless=True)
        self.team_id=team_id
        self.team_history=[]
        self.phi=phi
        self.psi=psi
        
    def getTeamHistory(self):
        for i in range(1,4):
            url=f"https://www.sofascore.com/api/v1/team/{self.team_id}/events/last/{i}"
            response=self.getUrlResponse(self.driver,url)
            if "events" in response.keys():
                events:list[dict]=response["events"]
                for event in events:
                    
                    tournament:dict=event.get("tournament")
                    league_info:dict=tournament.get("uniqueTournament")
                    league_id=league_info.get("id")
                    league=league_info.get("name")
                    season= event.get("season")
                    home_team_info:dict=event.get("homeTeam")
                    home_team=home_team_info.get("name")
                    home_id=home_team_info.get("id")
                    
                    #------------away team----------------
                    away_team_info:dict=event.get("awayTeam")
                    away_team=away_team_info.get("name")
                    away_id=away_team_info.get("id")
                    
                    home_score=event["homeScore"].get("normaltime")
                    away_score=event["awayScore"].get("normaltime")
                    
                    date=event.get("startTimestamp")
                    fm_date=datetime.fromtimestamp(date)
                    date_today=datetime.today()
                    days_diff=(date_today-fm_date).days
                    
                    match_id=event.get("id")
                    
                    line_up=self.getLineUp(match_id)
                    if days_diff<= self.psi*365 and line_up:
                        match_obj={
                            "league":league,
                            "league_id":league_id,
                            "home_team":home_team,
                            "home_id":home_id,
                            "home_score":home_score,
                            "away_team":away_team,
                            "away_id":away_id,
                            "away_score":away_score,
                            "match_date":fm_date.strftime("%Y-%m-%d"),
                            "match_id":match_id,
                            "season_id":season,
                            "home_formation":line_up["home_formation"],
                            "away_formation":line_up["away_formation"],
            
                        }
                        
                        home_players=self.getPlayersData(line_up["home_players"],days_diff,True,league,league_id,fm_date)
                        away_players=self.getPlayersData(line_up["away_players"],days_diff,False,league,league_id,fm_date)
                        match_obj["home_players"]=home_players
                        match_obj["away_players"]=away_players
                       
                        self.team_history.append(match_obj)
                        
                print(f"{i} history done")
            elif "error" in response.keys():
                print(f"History error{response}")
            
            time.sleep(2)
        
        self.driver.close()
    def getPlayersData(self,players:list[dict],days_diff:int,is_home:bool,league:str,league_id:int,match_date:datetime) -> list[dict]:
       
        weight=self.compute_weight(days_diff)
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
                "league":league,
                "league_id":league_id,
                "match_date":match_date.strftime("%Y-%m-%d"),
                "days_to_match":days_diff,
                "weight":weight,
                "substitute":substitute,
                "minutes_played":minutes_played,
                "position":player.get("standard_position"),
                "specific_position":player.get("specific_position"),
                "statistics":player.get("statistics")
            }
            players_data.append(c_player_obj)
            
        return players_data
            
    def getLineUp(self,match_id) -> dict|None:
        url=f"https://www.sofascore.com/api/v1/event/{match_id}/lineups"         
        response=self.getUrlResponse(self.driver,url)
        if "confirmed" in response.keys():
            home_players=response["home"]["players"]
            away_players=response["away"]["players"]
            home_formation=response["home"]["formation"]
            away_formation=response["away"]["formation"]
            
            player_mapper=AutomaticPositionMapper()
            hm_players=player_mapper.assign_positions(home_players,home_formation)
            aw_players=player_mapper.assign_positions(away_players,away_formation)
            
            
            return {"home_players":hm_players,
                    "away_players":aw_players,
                    "home_formation":home_formation,
                    "away_formation":away_formation
                    }
        else:
            print(f"Lineup Error - {response}")
   
    def compute_weight(self,days_diff):
        return np.exp(-self.phi * (days_diff / 3.5))
