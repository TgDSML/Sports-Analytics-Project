import pandas as pd

# 1. Διαβάζουμε το τελικό αρχείο κατοχής που έβγαλε το pipeline
possession_df = pd.read_csv("outputs/possession_30s_720p.csv")

# 2. Φιλτράρουμε τις γραμμές όπου όντως κάποια ομάδα είχε την μπάλα (διώχνουμε τα None)
active_possession = possession_df[possession_df['team'].isin(['Team A', 'Team B'])].copy()

# 3. Βρίσκουμε πότε αλλάζει ο παίκτης που έχει την μπάλα
# Το .shift() κοιτάζει τον παίκτη του προηγούμενου frame
player_changed = active_possession['nearest_player_id'] != active_possession['nearest_player_id'].shift()
team_changed = active_possession['team'] != active_possession['team'].shift()

# Κρατάμε μόνο τα frames όπου άλλαξε ο παίκτης (δηλαδή έγινε μεταβίβαση της μπάλας)
ball_transfers = active_possession[player_changed].dropna(subset=['nearest_player_id'])

# Αν άλλαξε ο παίκτης αλλά η ομάδα παρέμεινε η ίδια -> Επιτυχημένη Πάσα
# Αν άλλαξε ο παίκτης ΚΑΙ άλλαξε η ομάδα -> Λανθασμένη Πάσα / Κλέψιμο
# 4. Διαχωρίζουμε σε επιτυχημένες και λανθασμένες πάσες χρησιμοποιώντας το index του ball_transfers
success_passes = ball_transfers[~team_changed.reindex(ball_transfers.index)]
failed_passes = ball_transfers[team_changed.reindex(ball_transfers.index)]

success = len(success_passes)
failed = len(failed_passes)

# 5. Εκτύπωση των πραγματικών αποτελεσμάτων
print("--- ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΑΛΥΣΗΣ ΠΑΣΩΝ (FULL PIPELINE) ---")
print(f"Επιτυχημένες Πάσες: {success}")
print(f"Λανθασμένες Πάσες (Κλεψίματα/Λάθη): {failed}")

total = success + failed
if total > 0:
    accuracy = (success / total) * 100
    print(f"Ακρίβεια Πασών: {accuracy:.2f}%")
else:
    print("Δεν εντοπίστηκαν πάσες στο αρχείο κατοχής.")