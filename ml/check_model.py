import joblib

model = joblib.load("models/caregiver_model.joblib")
features = joblib.load("models/selected_features.joblib")

print("Model loaded successfully!")
print("Selected features:", features)