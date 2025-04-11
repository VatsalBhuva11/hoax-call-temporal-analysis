import csv
from collections import defaultdict

# Define year ranges
def get_range(year):
    year = int(year)
    if 2011 <= year <= 2015:
        return "2011-2015"
    elif 2016 <= year <= 2020:
        return "2016-2020"
    elif 2021 <= year <= 2025:
        return "2021-2025"
    else:
        return "Other"

# Initialize dictionary: counts[label][range] = count
counts = defaultdict(lambda: defaultdict(int))

count = 0
# Read the dataset with year as second column
with open('prcsd_with_date.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # skip header

    for row in reader:
        label = row[0]
        year = row[1]
        print(f"count: {count}")
        year_range = get_range(year)
        counts[label][year_range] += 1
        count += 1

# Print the counts
for label in counts:
    print(f"\n{label.upper()} calls:")
    for year_range in ["2011-2015", "2016-2020", "2021-2025"]:
        print(f"  {year_range}: {counts[label][year_range]}")
