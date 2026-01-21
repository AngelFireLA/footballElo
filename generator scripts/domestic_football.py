import pandas as pd
import numpy as np
from numba import jit
import time
import os


@jit(nopython=True)
def elo_update(ra, rb, score, K=40):
    """
    Numba-accelerated ELO rating update function.
    Fixed to avoid duplicate calculations.
    """
    # Calculate expected score for team A
    Ea = 1 / (1 + 10 ** ((rb - ra) / 400))

    # Update ratings based on actual score vs expected score
    # Score: 1.0 if A wins, 0.0 if B wins (A loses), 0.5 if Draw
    ra_new = ra + K * (score - Ea)
    rb_new = rb + K * ((1 - score) - (1 - Ea))
    return ra_new, rb_new


def main():
    start_time = time.time()
    print("Loading data...")

    # --- Configuration ---
    data_file_path = r'C:\Dev\Python\footballElo\processed_databases\Domestic Football results from 1888 to 2019\football_results.csv'
    output_file = 'elo_arpad_improved_domestic_football.csv'

    top_league_countries = ['England', 'Spain', 'Germany', 'Italy', 'France', 'Portugal', 'Netherlands']

    INITIAL_ELO = 1000.0
    TOTAL_BOOTSTRAP_PASSES = 100
    BOOTSTRAP_K = 80
    FINAL_K = 120
    MIN_MATCHES_THRESHOLD = 50
    # --- End Configuration ---

    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}")
        return

    try:
        df = pd.read_csv(data_file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    print("Preprocessing data...")
    required_cols = ['home_ident', 'away_ident', 'date', 'gh', 'ga']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing required columns. Found: {df.columns}. Required: {required_cols}")
        return

    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
    except Exception as e:
        print(f"Error converting 'date' column to datetime: {e}")
        return

    pattern = "|".join([f"({country})" for country in top_league_countries])
    all_clubs = set(df['home_ident']).union(set(df['away_ident']))
    filtered_clubs = {
        club for club in all_clubs
        if isinstance(club, str) and pd.Series([club]).str.contains(pattern, regex=True).any()
    }

    if not filtered_clubs:
        print("Error: No clubs found matching the specified countries.")
        return

    print(f"Found {len(filtered_clubs)} clubs in selected countries.")

    df = df[df['home_ident'].isin(filtered_clubs) & df['away_ident'].isin(filtered_clubs)].copy()
    # Sort data initially for the first forward pass and the final calculation pass
    df.sort_values('date', inplace=True)
    # Create a reversed view/copy for backward passes (more efficient than slicing each time)
    df_reversed = df.iloc[::-1].copy()

    dates = sorted(df['date'].unique())
    total_matches_processed = df.shape[0]
    print(f"Total matches to process involving these clubs: {total_matches_processed}")
    if total_matches_processed == 0:
        print("Error: No matches found involving clubs from the specified countries.")
        return

    # Track first and last appearance dates for each club
    first_appearance = {club: df['date'].max() for club in filtered_clubs}
    last_appearance = {club: df['date'].min() for club in filtered_clubs}
    for _, row in df.iterrows():
        home, away = row.home_ident, row.away_ident
        if home in filtered_clubs:
            first_appearance[home] = min(first_appearance[home], row.date)
            last_appearance[home] = max(last_appearance[home], row.date)
        if away in filtered_clubs:
            first_appearance[away] = min(first_appearance[away], row.date)
            last_appearance[away] = max(last_appearance[away], row.date)

    # --- PHASE 1: Alternating Bootstrap ---
    print(f"\nPHASE 1: Alternating Bootstrap ({TOTAL_BOOTSTRAP_PASSES} passes, K={BOOTSTRAP_K})...")
    # Initialize ELO ratings
    current_elo_dict = {club: INITIAL_ELO for club in filtered_clubs}

    # Track active clubs during each time period
    active_clubs = {club: False for club in filtered_clubs}

    for pass_num in range(1, TOTAL_BOOTSTRAP_PASSES + 1):
        pass_start_time = time.time()
        # Determine direction: Odd passes are forward, Even passes are backward
        is_forward_pass = (pass_num % 2 != 0)

        if is_forward_pass:
            print(f"  Starting Forward Pass {pass_num}/{TOTAL_BOOTSTRAP_PASSES}...")
            iterator = df.itertuples(index=False)
        else:
            print(f"  Starting Backward Pass {pass_num}/{TOTAL_BOOTSTRAP_PASSES}...")
            iterator = df_reversed.itertuples(index=False)

        # Create a temporary dict for the next state to avoid modifying while iterating
        next_elo_dict = current_elo_dict.copy()

        for row in iterator:
            home, away = row.home_ident, row.away_ident

            # Activate clubs when they appear in a match
            active_clubs[home] = True
            active_clubs[away] = True

            # Get current ratings from the *start* of the pass
            home_elo = current_elo_dict[home]
            away_elo = current_elo_dict[away]

            home_goals, away_goals = row.gh, row.ga
            if home_goals > away_goals:
                result = 1.0
            elif home_goals < away_goals:
                result = 0.0
            else:
                result = 0.5

            # Update ELO ratings using the bootstrap K
            new_home_elo, new_away_elo = elo_update(home_elo, away_elo, result, K=BOOTSTRAP_K)

            # Store updated ratings in the temporary dictionary for the *next* state
            next_elo_dict[home] = new_home_elo
            next_elo_dict[away] = new_away_elo

        # After iterating through all matches in the pass, update the main dictionary
        current_elo_dict = next_elo_dict
        pass_duration = time.time() - pass_start_time
        print(f"    Pass {pass_num} completed in {pass_duration:.2f} seconds.")

        # Show a sample team's progress
        if 'Real Madrid (Spain)' in current_elo_dict:
            print(f"      Real Madrid ELO after pass {pass_num}: {current_elo_dict['Real Madrid (Spain)']:.2f}")

    # --- PHASE 2: Final Calculation ---
    print(f"\nPHASE 2: Final Calculation (1 forward pass, K={FINAL_K})...")

    # Initialize team metrics
    elo_dict_final = {club: INITIAL_ELO for club in filtered_clubs}
    elo_history = {}  # Will store {date: {team: elo}}
    matches_played = {club: 0 for club in filtered_clubs}
    team_activity = {club: [] for club in filtered_clubs}  # Track active periods

    # Use the original sorted DataFrame for the final forward pass
    current_date = None
    daily_updates = {}

    for row in df.itertuples(index=False):
        home, away = row.home_ident, row.away_ident
        date = row.date

        # If we're on a new date, save the previous date's updates
        if current_date is not None and date != current_date:
            elo_history[current_date] = daily_updates.copy()
            daily_updates = {}
        current_date = date

        home_goals, away_goals = row.gh, row.ga
        if home_goals > away_goals:
            result = 1.0
        elif home_goals < away_goals:
            result = 0.0
        else:
            result = 0.5

        # Get current ratings before update for this match
        home_elo_before = elo_dict_final[home]
        away_elo_before = elo_dict_final[away]

        # Update ELO ratings using the final K
        new_home_elo, new_away_elo = elo_update(home_elo_before, away_elo_before, result, K=FINAL_K)
        elo_dict_final[home] = new_home_elo
        elo_dict_final[away] = new_away_elo

        # Record ratings *after* the match
        daily_updates[home] = new_home_elo
        daily_updates[away] = new_away_elo

        # Update matches played count
        matches_played[home] += 1
        matches_played[away] += 1

        # Record team activity
        team_activity[home].append(date)
        team_activity[away].append(date)

    # Save last date
    if current_date is not None:
        elo_history[current_date] = daily_updates.copy()

    # --- Post-Processing and Output ---
    print("\nPost-processing and creating final dataframe...")
    print(f"Filtering clubs with fewer than {MIN_MATCHES_THRESHOLD} matches played.")
    clubs_meeting_threshold = {
        club for club in filtered_clubs
        if matches_played.get(club, 0) >= MIN_MATCHES_THRESHOLD
    }
    print(f"Kept {len(clubs_meeting_threshold)} clubs out of {len(filtered_clubs)} initial clubs.")

    if not clubs_meeting_threshold:
        print("Error: No clubs met the minimum match threshold. Try lowering MIN_MATCHES_THRESHOLD.")
        return

    # Create a dataframe with dates as index
    elo_df = pd.DataFrame(index=dates)

    # For each club, create a series of their ratings at each date they played
    for club in clubs_meeting_threshold:
        club_ratings = {}
        # Only include dates when the team actually played
        club_dates = sorted(date for date in elo_history.keys() if club in elo_history[date])
        for date in club_dates:
            if club in elo_history[date]:
                club_ratings[date] = elo_history[date][club]

        # Convert to series and add to dataframe
        if club_dates:  # Only if the club played at least one match
            series = pd.Series(club_ratings)
            elo_df[club] = series

    # Fill gaps in ratings - but only within a club's active period
    for club in clubs_meeting_threshold:
        # Only fill forward within the period when the team was active
        start_date = first_appearance.get(club)
        end_date = last_appearance.get(club)

        if pd.notna(start_date) and pd.notna(end_date):
            # Create a mask for the active period
            mask = (elo_df.index >= start_date) & (elo_df.index <= end_date)
            # Only forward fill within this period
            elo_df.loc[mask, club] = elo_df.loc[mask, club].fillna(method='ffill')

            # Make sure the first rating in the active period exists
            first_valid_idx = elo_df.loc[mask, club].first_valid_index()
            if first_valid_idx is None:
                # If no valid rating in the active period, use the initial rating
                elo_df.loc[mask, club] = INITIAL_ELO
            elif first_valid_idx > start_date:
                # If first valid rating is after start date, backfill to start
                elo_df.loc[elo_df.index.to_series().between(start_date, first_valid_idx), club] = elo_df.loc[
                    first_valid_idx, club]

    # Round ratings to integers
    elo_df = elo_df.round().astype('Int64')  # Use Int64 to preserve NaN values

    print(f"Saving ELO history to {output_file}...")
    try:
        elo_df.to_csv(output_file)
        print(f"Successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

    # Print stats for a sample team
    sample_team = 'Real Madrid (Spain)'
    if sample_team in elo_df.columns:
        # Get the team's first and last ratings
        first_valid = elo_df[sample_team].first_valid_index()
        last_valid = elo_df[sample_team].last_valid_index()

        if first_valid is not None and last_valid is not None:
            first_elo = elo_df.loc[first_valid, sample_team]
            last_elo = elo_df.loc[last_valid, sample_team]
            match_count = matches_played.get(sample_team, 0)

            print(f"\nStats for '{sample_team}':")
            print(f"  First appearance: {first_appearance.get(sample_team)}")
            print(f"  Last appearance: {last_appearance.get(sample_team)}")
            print(f"  Matches played: {match_count}")
            print(f"  Initial ELO: {first_elo}")
            print(f"  Final ELO: {last_elo}")
            print(f"  ELO change: {last_elo - first_elo}")
    elif sample_team in filtered_clubs:
        print(f"'{sample_team}' was filtered out due to low match count (<{MIN_MATCHES_THRESHOLD}).")
    else:
        print(f"'{sample_team}' was not found in the initial country filter.")

    elapsed_time = time.time() - start_time
    print(f"\nExecution completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()