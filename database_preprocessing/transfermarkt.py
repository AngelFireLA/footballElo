import pandas as pd

# Load the CSV files
games_df = pd.read_csv('D:\Dev\Python/footballElo\original_databases/transfermarkt\games.csv')
competitions_df = pd.read_csv('D:\Dev\Python/footballElo\original_databases/transfermarkt\competitions.csv')
clubs_df = pd.read_csv('D:\Dev\Python/footballElo\original_databases/transfermarkt\clubs.csv')

# Drop unnecessary columns from games_df
columns_to_drop = [
    'game_id', 'season', 'round', 'home_club_position', 'away_club_position',
    'home_club_manager_name', 'away_club_manager_name', 'stadium', 'attendance', 'referee',
    'url', 'home_club_formation', 'away_club_formation', 'aggregate',
    'home_club_id', 'away_club_id'  # Removing club ID columns as names are already present
]
games_df_cleaned = games_df.drop(columns=columns_to_drop)

# Creating a dictionary to map competition IDs to names
competition_dict = competitions_df.set_index('competition_id')['name'].to_dict()
# Replacing competition_id with competition names
games_df_cleaned['competition_id'] = games_df_cleaned['competition_id'].map(competition_dict)

# Optionally, save the cleaned data to a new CSV file
games_df_cleaned.to_csv('cleaned_games.csv', index=False)
