import pandas as pd
import numpy as np
from numba import jit

@jit(nopython=True)
def elo_update(ra, rb, score, K=40):
    """
    Numba-accelerated ELO rating update function.
    """
    Ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    Eb = 1 / (1 + 10 ** ((ra - rb) / 400))
    ra_new = ra + K * (score - Ea)
    rb_new = rb + K * (1 - score - Eb)
    return ra_new, rb_new

def main():
    # Read data
    df = pd.read_csv(r'D:\Dev\Python\footballElo\processed_databases\Global football results for domestic leagues from 1993 to 2024\football_results.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')


    # Initialize clubs and ratings
    clubs = set(df['HomeTeam']).union(set(df['AwayTeam']))
    filtered_clubs = clubs
    elo_dict = {club: 1000 for club in filtered_clubs}
    dates = sorted(df['Date'].unique())
    elo_history = {date: {club: np.nan for club in filtered_clubs} for date in dates}

    # Sort data by date
    df.sort_values('Date', inplace=True)
    k = 80

    total_matches = df.shape[0]
    print(f"Total matches to process: {total_matches}")

    # Process each match
    for idx, row in enumerate(df.itertuples(index=False), 1):
        home, away = row.HomeTeam, row.AwayTeam
        if home not in filtered_clubs and away not in filtered_clubs:
            continue

        home_goals, away_goals = row.home_goal, row.away_goal
        date = row.Date

        if home_goals > away_goals:
            result = 1.0  # Home win
        elif home_goals < away_goals:
            result = 0.0  # Away win
        else:
            result = 0.5  # Draw

        # Update ELO ratings
        if home in filtered_clubs and away in filtered_clubs:
            new_home_elo, new_away_elo = elo_update(elo_dict[home], elo_dict[away], result, K=k)
            new_home_elo, new_away_elo = int(new_home_elo), int(new_away_elo)
            elo_dict[home] = new_home_elo
            elo_dict[away] = new_away_elo
            elo_history[date][home] = new_home_elo
            elo_history[date][away] = new_away_elo

        if idx % 1000 == 0:
            print(f"Processed {idx}/{total_matches} matches")

    # Create a DataFrame from the elo_history
    elo_df = pd.DataFrame.from_dict(elo_history, orient='index', columns=sorted(filtered_clubs))
    elo_df.fillna(method='ffill', inplace=True)  # Forward fill to carry previous ratings forward
    elo_df.to_csv(f'elo_arpad_static_global_football_results_{k}.csv')

if __name__ == "__main__":
    main()
