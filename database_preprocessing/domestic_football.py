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
    'home', 'away', 'home_country', 'away_country',
    'home_code', 'away_code', 'home_continent',
    'away_continent', 'continent', 'level'
]

# Example call (commented out to prevent execution)
remove_columns_from_csv('D:/Dev/Python/footballElo/original_databases/Domestic Football results from 1888 to 2019/football_results.csv', 'football_results_modified.csv', columns_to_remove)



