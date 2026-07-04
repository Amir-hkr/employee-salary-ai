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

from sklearn.model_selection import train_test_split

# Features (X)
X = df.drop("Salary", axis=1)

# Target (y)
y = df["Salary"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)