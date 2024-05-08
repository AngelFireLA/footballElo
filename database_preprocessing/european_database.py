import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('D:\Dev\Python/footballElo\original_databases\european databse 2008 to 2016\database.sqlite')

# SQL query to fetch and join the necessary tables
query = """
SELECT
    C.name AS country_name,
    L.name AS league_name,
    M.season,
    M.date,
    HT.team_long_name AS home_team_name,
    AT.team_long_name AS away_team_name,
    M.home_team_goal,
    M.away_team_goal
FROM Match AS M
JOIN Country AS C ON M.country_id = C.id
JOIN League AS L ON M.league_id = L.id
JOIN Team AS HT ON M.home_team_api_id = HT.team_api_id
JOIN Team AS AT ON M.away_team_api_id = AT.team_api_id;
"""

# Execute the query and load the data into a pandas DataFrame
match_data = pd.read_sql_query(query, conn)

# Closing the connection to the database
conn.close()


match_data.to_csv('match_data.csv', index=False)
