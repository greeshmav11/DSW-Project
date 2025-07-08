# Shows missing values per column (before cleaning)
print(merged_df.isnull().sum())

# Removes all rows that have at least one missing value 
merged_df = merged_df.dropna()  

# Shows missing values per column (after cleaning)
print(merged_df.isnull().sum())  


# Print the number of rows before filtering
print("Before:", len(merged_df))

# Keep only rows where the 'Label' column is 0 or 1
merged_df = merged_df[merged_df['Label'].isin([0, 1])]

# Print the number of rows after filtering
print("After:", len(merged_df))

merged_df.to_csv('data/cleaned_merged_news_market.csv', index=False)
