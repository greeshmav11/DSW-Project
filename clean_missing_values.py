print(merged_df.isnull().sum())  # Shows missing values per column (before cleaning)
merged_df = merged_df.dropna()  # Removes all rows that have at least one missing value 
print(merged_df.isnull().sum())  # Shows missing values per column (after cleaning)
