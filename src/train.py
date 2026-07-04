import pandas as pd

# Read cleaned dataset
df = pd.read_csv("data/cleaned_salary_data.csv")

# Remove Job Title
df = df.drop("Job Title", axis=1)

print(df.head())

print("\nColumns:")
print(df.columns)


from sklearn.preprocessing import LabelEncoder

# Create LabelEncoders
gender_encoder = LabelEncoder()
education_encoder = LabelEncoder()

# Encode categorical columns
df["Gender"] = gender_encoder.fit_transform(df["Gender"])
df["Education Level"] = education_encoder.fit_transform(df["Education Level"])

print("\nEncoded Dataset:")
print(df.head())