import numpy as np

def detect_passes(players_df, ball_df, team_assignments, scale_factor=1.5):
    successful_passes = 0
    failed_passes = 0
    current_possessor = None
    current_team = None
    
    frames = sorted(list(set(ball_df['frame']).intersection(set(players_df['frame']))))
    
    for frame in frames:
        ball_row = ball_df[ball_df['frame'] == frame]
        if ball_row.empty:
            continue
            
        bx, by = ball_row['center_x'].values[0], ball_row['center_y'].values[0]
        players_frame = players_df[players_df['frame'] == frame]
        
        closest_player = None
        min_dist = float('inf')
        dynamic_d_min = 0 
        
        for _, player in players_frame.iterrows():
            px, py = player['center_x'], player['center_y']
            dist = np.sqrt((bx - px)**2 + (by - py)**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_player = int(player['track_id'])
                
                # Παίρνουμε το ύψος του κουτιού (bounding box height).
                # Αν η στήλη λέγεται διαφορετικά στο CSV σας (π.χ. bbox_h), 
                # αλλάξτε το 'height' παρακάτω. Αν δεν βρει τη στήλη, βάζει 60 by default.
                player_h = player.get('height', 60)
                
                # Ορίζουμε το δυναμικό όριο ως [Ύψος Παίκτη] * [Συντελεστής]
                dynamic_d_min = player_h * scale_factor
                
        # Συγκρίνουμε την απόσταση με το δυναμικό όριο του συγκεκριμένου παίκτη
        if min_dist < dynamic_d_min:
            player_team = team_assignments.get(closest_player, "Unknown")
            if current_possessor is not None and closest_player != current_possessor:
                if player_team == current_team and player_team != "Unknown":
                    successful_passes += 1
                elif player_team != "Unknown" and current_team != "Unknown":
                    failed_passes += 1
                    
            current_possessor = closest_player
            current_team = player_team
            
    return successful_passes, failed_passes