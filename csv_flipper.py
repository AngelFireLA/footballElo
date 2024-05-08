import pandas as pd

# Load the CSV file
df = pd.read_csv('elo_arpad_static_international_results_80.csv')

# Transpose the dataframe
transposed_df = df.T

# Save the transposed dataframe to a new CSV file
transposed_df.to_csv('elo_arpad_static_international_results_80_flipped.csv', header=False)
