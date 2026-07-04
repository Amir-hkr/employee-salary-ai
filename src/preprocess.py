import pandas as pd

# Read dataset
df = pd.read_csv("data/Salary Data.csv")

# Show first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Dataset shape
print("\nShape:")
print(df.shape)

# Count missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Count duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

# Show new shape
print("\nNew Shape:")
print(df.shape)

# Check again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows After Cleaning:")
print(df.duplicated().sum())

# Save cleaned dataset
df.to_csv("data/cleaned_salary_data.csv", index=False)

print("\nCleaned dataset saved successfully!")