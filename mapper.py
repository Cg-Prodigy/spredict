class AutomaticPositionMapper:
    def __init__(self):
        self.formation_position_templates = {
            # 4-3-3 formations
            '4-3-3': {
                'GK': [0],
                'DEF': ['RB', 'RCB', 'LCB', 'LB'],  # Right to left
                'MID': ['RCM', 'CM', 'LCM'],  # Right to left  
                'FWD': ['RW', 'ST', 'LW']  # Right to left
            },
            '4-1-2-3': {
                'GK': [0],
                'DEF': ['RB', 'RCB', 'LCB', 'LB'],
                'MID': ['CDM', 'RCM', 'LCM'],
                'FWD': ['RW', 'ST', 'LW']
            },
            # 4-4-2 formations
            '4-4-2': {
                'GK': [0],
                'DEF': ['RB', 'RCB', 'LCB', 'LB'],
                'MID': ['RM', 'RCM', 'LCM', 'LM'],
                'FWD': ['RS', 'LS']  # Right striker, Left striker
            },
            '4-4-1-1': {
                'GK': [0],
                'DEF': ['RB', 'RCB', 'LCB', 'LB'],
                'MID': ['RM', 'RCM', 'LCM', 'LM'],
                'FWD': ['CF', 'ST']  # Center forward, Striker
            },
            '4-1-3-2': {
                'GK': [0],
                'DEF': ['RB', 'RCB', 'LCB', 'LB'],
                'MID': ['CDM', 'RM', 'CM', 'LM'],
                'FWD': ['RS', 'LS']
            },
            # 4-2-3-1 formations
            '4-2-3-1': {
                'GK': [0],
                'DEF': ['RB', 'RCB', 'LCB', 'LB'],
                'MID': ['CDM', 'CDM', 'RM', 'CAM', 'LM'],
                'FWD': ['ST']
            },
            '4-3-2-1': {
                'GK': [0],
                'DEF': ['RB', 'RCB', 'LCB', 'LB'],
                'MID': ['CDM', 'RCM', 'LCM', 'RAM', 'LAM'],
                'FWD': ['ST']
            },
            # 3-5-2 formations
            '3-5-2': {
                'GK': [0],
                'DEF': ['RCB', 'CB', 'LCB'],
                'MID': ['RWB', 'RCM', 'CM', 'LCM', 'LWB'],
                'FWD': ['RS', 'LS']
            },
            '3-4-3': {
                'GK': [0],
                'DEF': ['RCB', 'CB', 'LCB'],
                'MID': ['RWB', 'RCM', 'LCM', 'LWB'],
                'FWD': ['RW', 'ST', 'LW']
            },
            # 5-3-2 formations
            '5-3-2': {
                'GK': [0],
                'DEF': ['RWB', 'RCB', 'CB', 'LCB', 'LWB'],
                'MID': ['RCM', 'CM', 'LCM'],
                'FWD': ['RS', 'LS']
            },
            '5-4-1': {
                'GK': [0],
                'DEF': ['RWB', 'RCB', 'CB', 'LCB', 'LWB'],
                'MID': ['RM', 'RCM', 'LCM', 'LM'],
                'FWD': ['ST']
            }
        }
        
        # Map detailed positions to standard ones for interaction probabilities
        self.position_standardization = {
            # Goalkeeper
            'GK': 'GK',
            
            # Defenders
            'RB': 'RB', 'RWB': 'RB', 
            'LB': 'LB', 'LWB': 'LB',
            'RCB': 'CB', 'CB': 'CB', 'LCB': 'CB',
            
            # Midfielders  
            'CDM': 'CDM',
            'RCM': 'CM', 'CM': 'CM', 'LCM': 'CM',
            'RM': 'RM', 'LM': 'LM',
            'CAM': 'CAM', 'RAM': 'CAM', 'LAM': 'CAM',
            
            # Forwards
            'RW': 'RW', 'LW': 'LW',
            'RS': 'ST', 'ST': 'ST', 'LS': 'ST', 'CF': 'ST'
        }
    
    def detect_formation_pattern(self, players):
        def_count = len([p for p in players if p.get('position') == 'D'])
        mid_count = len([p for p in players if p.get('position') == 'M'])
        fwd_count = len([p for p in players if p.get('position') == 'F'])
        
        if def_count == 4:
            if mid_count == 3 and fwd_count == 3:
                return '4-3-3'
            elif mid_count == 4 and fwd_count == 2:
                return '4-4-2'
            elif mid_count == 5 and fwd_count == 1:
                return '4-2-3-1' if any('DM' in str(p.get('position', '')) for p in players) else '4-1-4-1'
        elif def_count == 3:
            if mid_count == 5 and fwd_count == 2:
                return '3-5-2'
            elif mid_count == 4 and fwd_count == 3:
                return '3-4-3'
        elif def_count == 5:
            if mid_count == 3 and fwd_count == 2:
                return '5-3-2'
            elif mid_count == 4 and fwd_count == 1:
                return '5-4-1'
                
        return '4-4-2'  # Default
    
    def assign_positions(self, players, formation=None):
        if not formation:
            formation = self.detect_formation_pattern(players)
        
        template = self.formation_position_templates.get(formation, self.formation_position_templates['4-4-2'])
        
        gk_players = [p for p in players if p.get('position') == 'G']
        def_players = [p for p in players if p.get('position') == 'D']
        mid_players = [p for p in players if p.get('position') == 'M']
        fwd_players = [p for p in players if p.get('position') == 'F']
        
        assigned_players = []
        
        # Assign goalkeeper
        if gk_players and template['GK']:
            gk_players[0]['specific_position'] = template['GK'][0]
            gk_players[0]['standard_position'] = self.position_standardization.get(template['GK'][0], 'GK')
            assigned_players.append(gk_players[0])
        
        # Assign defenders
        for i, player in enumerate(def_players):
            if i < len(template['DEF']):
                player['specific_position'] = template['DEF'][i]
                player['standard_position'] = self.position_standardization.get(template['DEF'][i], 'CB')
            else:
                player['specific_position'] = 'CB'
                player['standard_position'] = 'CB'
            assigned_players.append(player)
        
        # Assign midfielders
        for i, player in enumerate(mid_players):
            if i < len(template['MID']):
                player['specific_position'] = template['MID'][i]
                player['standard_position'] = self.position_standardization.get(template['MID'][i], 'CM')
            else:
                player['specific_position'] = 'CM'
                player['standard_position'] = 'CM'
            assigned_players.append(player)
        
        # Assign forwards
        for i, player in enumerate(fwd_players):
            if i < len(template['FWD']):
                player['specific_position'] = template['FWD'][i]
                player['standard_position'] = self.position_standardization.get(template['FWD'][i], 'ST')
            else:
                player['specific_position'] = 'ST'
                player['standard_position'] = 'ST'
            assigned_players.append(player)
        
        return assigned_players, formation