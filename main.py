import json
import time
from match_collector import MatchCollector
if __name__=="__main__":
    with open("json_data/all_matches/formatted_events.json","r") as file:
        events=json.loads(file.read())
    
    '''
    for each team get the:
        1. Get the lineup
        2. From the lineup 
            - get the players
                - for each player
                    - get their stats for the match
                    - calculate their weight
            - get the formation
    '''
    league=list(filter(lambda x:x['league']==679,events))
    count=0
    repeat=0
    league_list=[]
    for event in league:
        match_collector=MatchCollector(event)
        match_collector.getLineUp()
        results=match_collector.complete_match_obj
        if results:
            league_list.append(results)
            print(f"Done for match {event.get("match_id")}")
            count+=1
        else:
            repeat+=1
            
        if repeat>=50:
            break
        print(f"Moving to match - {count}")
        time.sleep(4)
        
        
    with open("json_data/league_info/league_679.json","w") as file:
        obj={
            "count":count,
            "data":league_list
        }
        json.dump(obj,file)
        
        
        