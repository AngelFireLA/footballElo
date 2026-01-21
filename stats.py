import pandas as pd
import matplotlib.pyplot as plt
import re

import os

elo_over_time_csv_absolute_path = r"C:\Dev\Python\footballElo\csv_files\global football results\elo_arpad_improved_global_football_50p_120k.csv"

def load_data(filename):
    return pd.read_csv(filename, index_col=0)

def compute_stats(df):
    stats = df.describe().transpose()
    stats['mean_rating'] = df.mean(axis=0)
    return stats[['mean', 'std', 'max', 'mean_rating']]

def plot_elo_ratings(df, teams):
    for team in teams:
        try:
            plt.figure(figsize=(10, 5))
            if team in df.columns:
                df[team].plot(title=f'ELO Ratings Over Time for {team}')
                plt.xlabel('Date')
                plt.ylabel('ELO Rating')
                team_name = team.replace("/", " ")
                team_name = re.sub(r'\\', ' ', team_name)
                #create elo_ratingss folder if not exist
                if not os.path.exists('elo_ratings'):
                    os.makedirs('elo_ratings')
                plt.savefig(f'elo_ratings/{team_name}_elo_ratings.png')
                plt.close()
        except OSError as E:
            print(E)

def top_teams(df, num_teams=1290):
    max_ratings = df.max()
    sorted_teams = max_ratings.sort_values(ascending=False)
    return sorted_teams.head(num_teams)

def main():
    df = load_data(elo_over_time_csv_absolute_path)
    stats = compute_stats(df)
    stats.to_csv('elo_ratings_stats.csv')
    print("computed stats")
    top_teams_stats = top_teams(df)
    print(top_teams_stats)

    top_teams_list = top_teams_stats.index.tolist()

    plot_elo_ratings(df, top_teams_list)
    print("plotted elo ratings")
    top_teams_stats.to_csv('top_teams.csv')
    print("saved top teams stats")

if __name__ == '__main__':
    main()
