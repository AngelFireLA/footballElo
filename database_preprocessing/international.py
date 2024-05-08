# Importing the required CSV module
import csv


# Function to remove specified columns from a CSV file and save the result to a new file
def remove_columns_from_csv(input_file_path, output_file_path, columns_to_remove):
    """
    Remove specified columns from a CSV file.

    Parameters:
    input_file_path (str): The path to the input CSV file.
    output_file_path (str): The path where the output CSV file should be saved.
    columns_to_remove (list): A list of column names to be removed from the CSV file.
    """
    # Opening the input file for reading
    with open(input_file_path, mode='r', newline='', encoding='utf-8') as infile:
        # Using DictReader to read the CSV file into a dictionary format
        reader = csv.DictReader(infile)
        # The remaining columns are the ones not listed in columns_to_remove
        remaining_columns = [col for col in reader.fieldnames if col not in columns_to_remove]

        # Opening the output file for writing
        with open(output_file_path, mode='w', newline='', encoding='utf-8') as outfile:
            # Using DictWriter to write the dictionary into a CSV format with the remaining columns
            writer = csv.DictWriter(outfile, fieldnames=remaining_columns)
            # Writing the header row
            writer.writeheader()

            # Writing the remaining rows
            for row in reader:
                writer.writerow({col: row[col] for col in remaining_columns})


# Columns to remove based on the user's request
columns_to_remove = [
    'country', 'neutral'
]

# Example call (commented out to prevent execution)
remove_columns_from_csv('D:\Dev\Python/footballElo\original_databases\international 1872 to now/all_matches.csv', 'football_results_modified.csv', columns_to_remove)

import pandas as pd

# Load the data
countries_names_path = r"D:\Dev\Python\footballElo\original_databases\international 1872 to now\countries_names.csv"
all_matches = pd.read_csv('football_results_modified.csv')
countries_names = pd.read_csv(countries_names_path)

# Create a mapping from original_name to current_name
name_map = dict(zip(countries_names['original_name'], countries_names['current_name']))

# Replace the home_team and away_team columns using the mapping
all_matches['home_team'] = all_matches['home_team'].map(name_map).fillna(all_matches['home_team'])
all_matches['away_team'] = all_matches['away_team'].map(name_map).fillna(all_matches['away_team'])

# Save the updated DataFrame to a new CSV file
all_matches.to_csv('updated_all_matches.csv', index=False)



