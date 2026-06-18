import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# 1. Φόρτωση των δεδομένων (Γίνεται μόνο ΜΙΑ φορά!)
possession_df = pd.read_csv("outputs/possession_30s_720p.csv")
debug_df = pd.read_csv("outputs/possession_debug_30s_720p.csv")

# 2. Ρυθμίσεις εμφάνισης για την κάθε ομάδα
teams_config = {
    'Team A': {'color': '#ff3366', 'font_color': 'white', 'filename': 'passing_network_team_a_tactical.png'},
    'Team B': {'color': '#00ffcc', 'font_color': 'black', 'filename': 'passing_network_team_b_tactical.png'}
}

# 3. Επανάληψη και για τις δύο ομάδες
for team_name, config in teams_config.items():


    # Εύρεση της μέσης θέσης (X, Y) κάθε παίκτη
    team_debug = debug_df[debug_df['team'] == team_name].dropna(subset=['nearest_player_id', 'nearest_player_center_x', 'nearest_player_center_y'])
    
    positions = {}
    for player_id, group in team_debug.groupby('nearest_player_id'):
        avg_x = group['nearest_player_center_x'].mean()
        avg_y = group['nearest_player_center_y'].mean()
        positions[int(player_id)] = (avg_x, avg_y)

    # Εξαγωγή των Πασών
    team_possession = possession_df[possession_df['team'] == team_name].dropna(subset=['nearest_player_id']).copy()
    
    # Αν η ομάδα δεν ακούμπησε καθόλου την μπάλα, προχωράμε στην επόμενη
    if team_possession.empty:
        print(f"Δεν βρέθηκε καθόλου κατοχή για την {team_name}.")
        continue

    team_possession['nearest_player_id'] = team_possession['nearest_player_id'].astype(int)

    passes = []
    player_changed = team_possession['nearest_player_id'] != team_possession['nearest_player_id'].shift()
    transfers = team_possession[player_changed]

    passers = transfers['nearest_player_id'].iloc[:-1].values
    receivers = transfers['nearest_player_id'].iloc[1:].values

    for passer, receiver in zip(passers, receivers):
        if passer != receiver:
            passes.append((passer, receiver))

    pass_counts = Counter(passes)

    # Δημιουργία Γραφήματος
    G = nx.DiGraph()
    for player_id in positions.keys():
        G.add_node(player_id)
    for (u, v), weight in pass_counts.items():
        G.add_edge(u, v, weight=weight)

    # Αν η ομάδα δεν έχει καθόλου παίκτες στο γράφημα, προχωράμε
    if len(G.nodes()) == 0:
        print(f"Δεν υπάρχουν αρκετά δεδομένα για να φτιαχτεί γράφημα για την {team_name}.")
        continue

    # Οπτικοποίηση
    plt.figure(figsize=(14, 9))
    plt.title(f"{team_name} - Passing Network (Tactical Layout)", fontsize=18, fontweight='bold', color='white')

    fig = plt.gcf()
    fig.patch.set_facecolor('#1a1a1a')
    ax = plt.gca()
    ax.set_facecolor('#1a1a1a')

    touches = team_possession['nearest_player_id'].value_counts()
    node_sizes = [touches.get(node, 1) * 60 for node in G.nodes()] 
    edge_weights = [G[u][v]['weight'] * 2.5 for u, v in G.edges()]

    tactical_positions = nx.spring_layout(G, k=0.5, iterations=50)
    
    nx.draw(G, pos=tactical_positions, with_labels=True, 
            node_color=config['color'], edge_color='#ffffff',
            node_size=node_sizes, width=edge_weights, 
            font_size=11, font_weight='bold', font_color=config['font_color'], 
            arrows=True, arrowsize=25, alpha=0.9, ax=ax)

    plt.text(0.98, 0.02, f"Total Successful Passes: {len(passes)}", 
             transform=ax.transAxes, color='gray', fontsize=12, 
             ha='right', va='bottom')

    output_path = f"outputs/{config['filename']}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

