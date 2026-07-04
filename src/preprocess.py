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


# ==========================
# Basic Data Analysis (EDA)
# ==========================

print("\n========== Statistics ==========")
print(df.describe())

print("\n========== Data Types ==========")
print(df.dtypes)

print("\n========== Gender ==========")
print(df["Gender"].value_counts())

print("\n========== Education Level ==========")
print(df["Education Level"].value_counts())

print("\n========== Top 10 Job Titles ==========")
print(df["Job Title"].value_counts().head(10))

print("\n========== Salary ==========")
print("Minimum Salary :", df["Salary"].min())
print("Maximum Salary :", df["Salary"].max())
print("Average Salary :", round(df["Salary"].mean(), 2))


print("\n========== Possible Outliers ==========")

outliers = df[df["Salary"] < 10000]

print(outliers)


# Remove salary outliers
df = df[df["Salary"] >= 10000]

print("\nShape After Removing Outliers:")
print(df.shape)

print("\nNew Minimum Salary:")
print(df["Salary"].min())

# Save cleaned dataset again
df.to_csv("data/cleaned_salary_data.csv", index=False)

print("\nDataset updated successfully!")