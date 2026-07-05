import joblib
import pandas as pd

# Load model
model = joblib.load("models/salary_model.pkl")

# Load encoders
gender_encoder = joblib.load("models/gender_encoder.pkl")
education_encoder = joblib.load("models/education_encoder.pkl")

# -----------------------------
# User Input
# -----------------------------
age = 30
gender = "Male"
education = "Master's"
experience = 5

# Encode categorical values
gender = gender_encoder.transform([gender])[0]
education = education_encoder.transform([education])[0]

# Create DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education Level": [education],
    "Years of Experience": [experience]
})

# Predict
salary = model.predict(input_data)

print("\nPredicted Salary:")
print(f"${salary[0]:,.2f}")