import pandas as pd
import numpy as np
from numba import jit
import time
import os

@jit(nopython=True)
def elo_update(ra, rb, score, K=40):
    """
    Numba-accelerated ELO rating update function.
    """
    Ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    ra_new = ra + K * (score - Ea)
    rb_new = rb + K * ((1 - score) - (1 - Ea)) # Equivalent to rb + K * (score_b - Eb)
    return ra_new, rb_new

def main():
    start_time = time.time()
    print("Loading data...")

    # --- Configuration ---
    data_file_path = r'C:\Dev\Python\footballElo\processed_databases\Global football results for domestic leagues from 1993 to 2024\football_results.csv'
    # Consider adding dataset name/year range to template if running variations
    output_file_template = 'elo_arpad_improved_global_football_{passes}p_K{bK}-{fK}_thr{thr}.csv'

    INITIAL_ELO = 1000.0
    # --- Parameter Tuning Considerations ---
    # Higher passes needed for stability, especially with higher K.
    TOTAL_BOOTSTRAP_PASSES = 200 # Increased passes for better stabilization
    # Bootstrap K: Higher values cause faster initial convergence but more volatility.
    # Lower values are more stable but require more passes. 80 is quite high.
    BOOTSTRAP_K = 60 # Slightly reduced K for potentially more stability
    # Final K: Determines sensitivity to recent results in the final calculation.
    # 120 is very high, leading to large rating swings. Common values are 20-60.
    FINAL_K = 40 # Reduced final K for less volatile final ratings
    MIN_MATCHES_THRESHOLD = 39 # Minimum matches played to be included in final output
    # --- End Configuration ---

    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}")
        return

    try:
        # Optimisation: Specify dtype for potentially faster loading
        dtypes = {'HomeTeam': 'str', 'AwayTeam': 'str', 'home_goal': 'float', 'away_goal': 'float'} # Use float for goals initially for NaN handling
        df = pd.read_csv(data_file_path, dtype=dtypes, usecols=['HomeTeam', 'AwayTeam', 'Date', 'home_goal', 'away_goal'])
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    print("Preprocessing data...")
    # Check required columns again after potentially limiting with usecols
    required_cols = ['HomeTeam', 'AwayTeam', 'Date', 'home_goal', 'away_goal']
    if not all(col in df.columns for col in required_cols):
         print(f"Error: Missing required columns after loading. Found: {df.columns}. Required: {required_cols}")
         return

    try:
        # Specify format for robustness, coerce errors to NaT
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        # Drop rows where critical information is missing BEFORE processing
        df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'home_goal', 'away_goal'], inplace=True)
    except Exception as e:
        print(f"Error processing 'Date' or dropping NaNs: {e}")
        return

    # Convert goals to integers after dropping NaNs
    try:
        df['home_goal'] = df['home_goal'].astype(np.int16) # Use smaller int type
        df['away_goal'] = df['away_goal'].astype(np.int16)
    except Exception as e:
        print(f"Error converting goal columns to integer: {e}")
        # You might choose to return here or proceed if some failed conversions are acceptable
        # df = df[pd.to_numeric(df['home_goal'], errors='coerce').notna()] # Example: Keep only rows where conversion is possible
        # df = df[pd.to_numeric(df['away_goal'], errors='coerce').notna()]

    # Identify all unique valid clubs
    all_clubs = set(df['HomeTeam']).union(set(df['AwayTeam']))
    # Ensure clubs are strings (though dropna above should handle most non-strings)
    filtered_clubs = {club for club in all_clubs if isinstance(club, str) and club} # Ensure non-empty strings

    if not filtered_clubs:
        print("Error: No valid club names found after preprocessing.")
        return

    print(f"Found {len(filtered_clubs)} unique clubs.")

    # Sort data by date is crucial for forward passes
    df.sort_values('Date', inplace=True)
    # Create a reversed copy for backward passes efficiently
    df_reversed = df.iloc[::-1].copy()

    dates = sorted(df['Date'].unique()) # Unique dates for the final DataFrame index
    total_matches_processed_in_df = df.shape[0]
    print(f"Total valid matches to process: {total_matches_processed_in_df}")
    if total_matches_processed_in_df == 0:
        print("Error: No matches remaining after preprocessing.")
        return

    # --- PHASE 1: Alternating Bootstrap ---
    print(f"\n--- PHASE 1: Alternating Bootstrap ({TOTAL_BOOTSTRAP_PASSES} passes, K={BOOTSTRAP_K}) ---")
    # Initialize ELO ratings dictionary
    current_elo_dict = {club: INITIAL_ELO for club in filtered_clubs}

    for pass_num in range(1, TOTAL_BOOTSTRAP_PASSES + 1):
        pass_start_time = time.time()
        is_forward_pass = (pass_num % 2 != 0)
        direction = "Forward" if is_forward_pass else "Backward"
        iterator_df = df if is_forward_pass else df_reversed

        print(f"  Starting {direction} Pass {pass_num}/{TOTAL_BOOTSTRAP_PASSES}...")

        # Use a temporary dictionary to store updates for this pass.
        # This prevents a match later in the pass from using ratings updated
        # by a match earlier *in the same pass*. Ratings are updated between passes.
        next_elo_dict = current_elo_dict.copy()
        matches_in_pass = 0

        # Use itertuples for performance
        for row in iterator_df.itertuples(index=False):
            # Get data directly using named tuple attributes
            home, away = row.HomeTeam, row.AwayTeam
            home_goals, away_goals = row.home_goal, row.away_goal

            # We already filtered clubs and dropped NaNs, but a check might be useful
            # if intermediate steps could reintroduce issues (unlikely here).
            # if home not in current_elo_dict or away not in current_elo_dict:
            #    continue # Should not happen if filtered_clubs is correct

            # Get ratings *at the start of the pass*
            home_elo = current_elo_dict[home]
            away_elo = current_elo_dict[away]

            # Determine score based on goals
            if home_goals > away_goals: result = 1.0
            elif home_goals < away_goals: result = 0.0
            else: result = 0.5

            # Calculate updated ELOs using Bootstrap K
            new_home_elo, new_away_elo = elo_update(home_elo, away_elo, result, K=BOOTSTRAP_K)

            # Store updates in the temporary dictionary
            next_elo_dict[home] = new_home_elo
            next_elo_dict[away] = new_away_elo
            matches_in_pass += 1

        # Update the main ELO dictionary *after* the pass is complete
        current_elo_dict = next_elo_dict
        pass_duration = time.time() - pass_start_time
        print(f"    Pass {pass_num} completed in {pass_duration:.2f} seconds ({matches_in_pass} matches processed).")

        # Optional: Monitor a specific team's progress during bootstrap
        sample_team_check = 'Real Madrid'
        if sample_team_check in current_elo_dict:
             print(f"      {sample_team_check} ELO after pass {pass_num}: {current_elo_dict[sample_team_check]:.2f}")

    print("--- Bootstrap Phase Completed ---")

    # --- PHASE 2: Final Calculation ---
    print(f"\n--- PHASE 2: Final Calculation (1 forward pass, K={FINAL_K}) ---")

    # Start final calculation with ratings stabilized by the bootstrap phase
    elo_dict_final = current_elo_dict.copy()
    elo_history = {}  # Stores {date: {team: elo_after_match_on_this_date}}
    matches_played = {club: 0 for club in filtered_clubs} # Track matches for filtering

    current_date = None
    daily_updates = {} # Temporarily store updates for the current date being processed
    final_matches_processed = 0

    # Iterate through the original, date-sorted DataFrame for the final forward pass
    for row in df.itertuples(index=False):
        home, away = row.HomeTeam, row.AwayTeam
        date = row.Date
        home_goals, away_goals = row.home_goal, row.away_goal

        # Check if the date has changed. If so, store the previous day's results.
        if current_date is not None and date != current_date:
            if daily_updates: # Only store if there were matches on that day
                 elo_history[current_date] = daily_updates.copy()
            daily_updates = {} # Reset for the new date
        current_date = date

        # Get ratings *before* this match (using the latest state from elo_dict_final)
        home_elo_before = elo_dict_final[home]
        away_elo_before = elo_dict_final[away]

        # Determine result
        if home_goals > away_goals: result = 1.0
        elif home_goals < away_goals: result = 0.0
        else: result = 0.5

        # Update ELO ratings using the Final K value
        new_home_elo, new_away_elo = elo_update(home_elo_before, away_elo_before, result, K=FINAL_K)

        # Update the main dictionary immediately for subsequent matches on the *same day*
        elo_dict_final[home] = new_home_elo
        elo_dict_final[away] = new_away_elo

        # Record the ratings *after* this match in the temporary daily storage
        # If a team plays twice on one day (rare), this stores the rating after the second match.
        daily_updates[home] = new_home_elo
        daily_updates[away] = new_away_elo

        # Increment match counts for final filtering
        matches_played[home] += 1
        matches_played[away] += 1
        final_matches_processed += 1

        # Progress indicator for the final pass
        if final_matches_processed % 50000 == 0:
             print(f"  Processed {final_matches_processed}/{total_matches_processed_in_df} matches in final calculation...")

    # Store the updates from the very last day
    if current_date is not None and daily_updates:
        elo_history[current_date] = daily_updates.copy()

    print(f"--- Final Calculation Phase Completed ({final_matches_processed} matches processed) ---")

    # --- Post-Processing and Output ---
    print("\n--- Post-Processing and Output ---")

    # Filter clubs based on the minimum number of matches played during the final pass
    print(f"Filtering clubs with fewer than {MIN_MATCHES_THRESHOLD} matches played.")
    clubs_meeting_threshold = {
        club for club in filtered_clubs
        if matches_played.get(club, 0) >= MIN_MATCHES_THRESHOLD
    }
    print(f"Kept {len(clubs_meeting_threshold)} clubs out of {len(filtered_clubs)} initial clubs based on threshold.")

    # *** Critical Fix: Use the filtered list ***
    if not clubs_meeting_threshold:
        print(f"Error: No clubs met the minimum match threshold ({MIN_MATCHES_THRESHOLD}). No output generated.")
        return
    final_clubs_to_output = sorted(list(clubs_meeting_threshold)) # Sort for consistent column order

    print(f"Preparing output DataFrame for {len(final_clubs_to_output)} clubs...")

    # Create the final DataFrame using all unique dates from the dataset as the index
    output_dates = sorted(df['Date'].unique())
    elo_df = pd.DataFrame(index=output_dates, columns=final_clubs_to_output)

    # Populate the DataFrame using the recorded history
    # This is more memory-intensive than iterating again but simpler to implement here
    print("Populating DataFrame from history...")
    for club in final_clubs_to_output:
        # Create a mapping of date -> elo for this specific club from the history
        club_ratings_over_time = {date: day_data.get(club) # Use .get() for safety
                                  for date, day_data in elo_history.items()
                                  if club in day_data}
        # Map these ratings onto the DataFrame index
        elo_df[club] = elo_df.index.map(club_ratings_over_time)

    print("Filling gaps in ELO ratings...")
    # 1. Forward Fill: Propagate the last known rating forward.
    #    This assumes a team's rating remains constant between their matches.
    elo_df.fillna(method='ffill', inplace=True)

    # 2. Backward Fill: Fill NaNs at the beginning of a team's history
    #    (before their first match in the dataset) with their *first calculated* rating.
    #    This is arguably better than filling with INITIAL_ELO if they start much higher/lower.
    elo_df.fillna(method='bfill', inplace=True)

    # 3. Fill Remaining NaNs: Any club that met the threshold but somehow still has NaNs
    #    (e.g., if their first/last match date had no rating recorded - very unlikely)
    #    or potentially clubs filtered out *before* history recording (shouldn't happen here).
    #    Fill these rare cases with INITIAL_ELO.
    elo_df.fillna(value=INITIAL_ELO, inplace=True)

    print("Rounding ratings...")
    # Round ratings to integers (or keep float if desired)
    try:
        elo_df = elo_df.round().astype(np.int32) # Use int32 for memory efficiency
    except Exception as e:
        print(f"Warning: Could not convert all ratings to integer. Keeping float. Error: {e}")
        elo_df = elo_df.round(2) # Keep as float with rounding

    # Construct final output filename
    output_file = output_file_template.format(
        passes=TOTAL_BOOTSTRAP_PASSES,
        bK=BOOTSTRAP_K,
        fK=FINAL_K,
        thr=MIN_MATCHES_THRESHOLD
    )
    print(f"Saving ELO history to {output_file}...")
    try:
        elo_df.to_csv(output_file)
        print(f"Successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

    # --- Optional: Print stats for a sample team ---
    # sample_team_final = 'Real Madrid'
    # if sample_team_final in elo_df.columns:
    #     print(f"\nStats for '{sample_team_final}':")
    #     print(f"  Matches counted: {matches_played.get(sample_team_final, 0)}")
    #     try:
    #         print(f"  First recorded ELO: {elo_df[sample_team_final].iloc[elo_df[sample_team_final].first_valid_index()]}")
    #         print(f"  Last recorded ELO: {elo_df[sample_team_final].iloc[-1]}")
    #     except IndexError:
    #         print("  Could not retrieve first/last ELO (IndexError).")
    # elif sample_team_final in filtered_clubs:
    #     print(f"\n'{sample_team_final}' did not meet the match threshold ({MIN_MATCHES_THRESHOLD}).")
    # else:
    #      print(f"\n'{sample_team_final}' was not found in the dataset.")


    elapsed_time = time.time() - start_time
    print(f"\nExecution completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()