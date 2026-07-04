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


from sklearn.ensemble import RandomForestRegressor

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("\nModel trained successfully!")


from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
import math

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

print("\n========== Model Evaluation ==========")
print(f"R2 Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")



import joblib

# Save trained model
joblib.dump(model, "models/salary_model.pkl")

print("\nModel saved successfully!")


# Save encoders
joblib.dump(gender_encoder, "models/gender_encoder.pkl")
joblib.dump(education_encoder, "models/education_encoder.pkl")

print("Encoders saved successfully!")