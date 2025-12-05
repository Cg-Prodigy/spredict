import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class InteractionModel:
    """
    Estimates interaction probabilities between players based on their positions.
    
    Since we do not have the granular event-level data (e.g., 'Player A tackled Player B') 
    required to train the multinomial models described in the paper, we use a 
    **Geometric Interaction Model**.
    
    Logic:
    - Players are mapped to (x, y) coordinates on the pitch.
    - Interaction probability is inversely proportional to the distance between 
      an attacking player and a defending player.
    - e.g., A Left Winger (LW) will be physically close to the opposing Right Back (RB),
      so their interaction weight will be high.
    """
    
    def __init__(self, decay_factor: float = 2.0):
        self.decay_factor = decay_factor
        self.is_trained = True # Heuristic model doesn't need training
        
        # Pitch coordinates (0-100 x, 0-100 y)
        # Home plays Left -> Right (0 -> 100)
        # Away plays Right -> Left (100 -> 0)
        # Formation-specific coordinates
        # We define offsets or overrides for standard positions based on formation
        self.formation_coordinates = {
            "4-4-2": {
                'CDM': (40, 50), 'CM': (50, 50), 'CAM': (60, 50),
                'LM': (50, 10), 'RM': (50, 90),
                'ST': (85, 50), 'SS': (75, 50)
            },
            "4-3-3": {
                'CDM': (40, 50), 
                'LCM': (50, 35), 'RCM': (50, 65),
                'LW': (80, 10), 'RW': (80, 90), 'ST': (85, 50)
            },
            "4-2-3-1": {
                'LDM': (40, 35), 'RDM': (40, 65),
                'LAM': (65, 20), 'CAM': (65, 50), 'RAM': (65, 80),
                'ST': (85, 50)
            },
            "3-5-2": {
                'LCB': (25, 30), 'CB': (20, 50), 'RCB': (25, 70),
                'LWB': (45, 10), 'RWB': (45, 90),
                'LDM': (40, 40), 'RDM': (40, 60), 'CAM': (60, 50),
                'ST': (85, 40), 'ST': (85, 60) # Two strikers
            },
            "3-4-3": {
                'LCB': (25, 30), 'CB': (20, 50), 'RCB': (25, 70),
                'LM': (50, 10), 'RM': (50, 90),
                'CM': (50, 40), 'CM': (50, 60),
                'LW': (80, 15), 'RW': (80, 85), 'ST': (85, 50)
            }
        }
        
        # Fallback base coordinates (Generic 4-4-2 ish)
        self.base_coordinates = {
            'GK': (5, 50),
            'LB': (30, 10), 'LCB': (25, 35), 'CB': (20, 50), 'RCB': (25, 65), 'RB': (30, 90),
            'CDM': (40, 50), 'LDM': (40, 35), 'RDM': (40, 65),
            'LM': (50, 10), 'LCM': (50, 35), 'CM': (50, 50), 'RCM': (50, 65), 'RM': (50, 90),
            'CAM': (65, 50), 'LAM': (65, 30), 'RAM': (65, 70),
            'LW': (80, 10), 'SS': (75, 50), 'ST': (85, 50), 'RW': (80, 90),
        }

    def train(self, historical_matches: List[Dict]):
        """
        Placeholder for training. 
        In a full implementation with event data, this would fit the multinomial logit models.
        For the geometric model, we don't need to train, but we keep the interface.
        """
        print("ℹ️ Using Geometric Interaction Model (No training required)")
        self.is_trained = True

    def get_player_coordinates(self, position: str, is_home_team: bool, formation: str = None) -> Tuple[float, float]:
        """Get (x, y) coordinates for a position, adjusted for team side and formation."""
        
        # 1. Try to get specific coordinates for this formation
        base_pos = None
        if formation and formation in self.formation_coordinates:
            base_pos = self.formation_coordinates[formation].get(position)
            
        # 2. Fallback to generic coordinates if not found
        if base_pos is None:
            base_pos = self.base_coordinates.get(position, (50, 50))
        
        if is_home_team:
            return base_pos
        else:
            # Mirror coordinates for away team (playing Right -> Left)
            return (100 - base_pos[0], 100 - base_pos[1])

    def calculate_interaction_weight(self, pos_a: str, pos_b: str, formation_a: str = None, formation_b: str = None) -> float:
        """Calculate interaction weight based on distance between positions."""
        # Assume A is Home, B is Away (or vice versa, distance is symmetric)
        coord_a = self.get_player_coordinates(pos_a, True, formation_a)
        coord_b = self.get_player_coordinates(pos_b, False, formation_b)
        
        dist = np.sqrt((coord_a[0] - coord_b[0])**2 + (coord_a[1] - coord_b[1])**2)
        
        # Inverse distance weighting
        # Add small epsilon to avoid division by zero
        weight = 1.0 / (dist**self.decay_factor + 100) 
        return weight

    def calculate_delta_start(self, home_players: List[Dict], away_players: List[Dict], 
                            home_formation: str, away_formation: str) -> float:
        """
        Calculate the weighted difference in AWS ratings (Δ_start).
        Formula: Σ Σ p_ij (AWS_i - AWS_j)
        """
        home_starters = [p for p in home_players if not p.get('substitute', False)]
        away_starters = [p for p in away_players if not p.get('substitute', False)]
        
        total_weighted_diff = 0.0
        
        # For each home player
        # For each home player
        for h_player in home_starters:
            # Prioritize specific_position (e.g., 'RCB'), then position (e.g., 'DEF'), then default
            h_pos = h_player.get('specific_position') or h_player.get('position', 'CM')
            h_aws = h_player.get('aws', 0)
            
            # Calculate interaction weights with all away players
            weights = []
            for a_player in away_starters:
                a_pos = a_player.get('specific_position') or a_player.get('position', 'CM')
                w = self.calculate_interaction_weight(h_pos, a_pos, home_formation, away_formation)
                weights.append(w)
            
            # Normalize weights to sum to 1 for this home player
            # (i.e., he spends 100% of his time interacting with SOMEONE)
            total_weight = sum(weights)
            if total_weight > 0:
                norm_weights = [w / total_weight for w in weights]
            else:
                norm_weights = [1.0 / len(away_starters)] * len(away_starters)
            
            # Calculate contribution to Δ
            # Σ p_ij * (AWS_i - AWS_j)
            player_contribution = 0
            for idx, a_player in enumerate(away_starters):
                a_aws = a_player.get('aws', 0)
                p_ij = norm_weights[idx]
                player_contribution += p_ij * (h_aws - a_aws)
                
            total_weighted_diff += player_contribution
            
        # Normalize by number of home starters (usually 11) to get average advantage per player
        if len(home_starters) > 0:
            total_weighted_diff /= len(home_starters)
            
        return total_weighted_diff
