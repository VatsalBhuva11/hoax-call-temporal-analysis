import pandas as pd

# Assuming your data is tab-separated and has columns: label, year, and word columns
# Read the data (adjust the path/filename as needed)
df = pd.read_csv('prcsd_with_date.csv', sep=',', header=None)

# Function to filter words in each row (keeping only words with length > 2)
def filter_short_words(row):
    return [word for word in row if not (isinstance(word, str) and len(word) <= 2)]

# Apply the filtering to each row (excluding the first two columns: label and year)
filtered_data = df.iloc[:, 2:].apply(filter_short_words, axis=1)

# Combine the filtered words with the original label and year columns
result = pd.concat([
    df.iloc[:, :2],  # Keep label and year columns
    filtered_data.apply(pd.Series)  # Convert filtered words back to columns
], axis=1)

# Save to a new file (tab-separated, without headers)
result.to_csv('filtered_data.tsv', sep='\t', header=False, index=False)