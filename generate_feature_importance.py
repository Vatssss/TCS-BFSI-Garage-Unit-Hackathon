import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load your original dataset
df = pd.read_csv("german_credit_data.csv")  # replace with your real preprocessed file

# Define features and target
X = df.drop("Risk", axis=1)
y = df["Risk"]

# Split and save
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")

# Load model and features
model = joblib.load("voting_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Check if model has feature_importances_
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
else:
    importances = np.mean([
        est.feature_importances_
        for est in model.estimators_
        if hasattr(est, 'feature_importances_')
    ], axis=0)

# Create DataFrame
feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Save
feat_df.to_csv("feature_importance.csv", index=False)
print("Feature importance saved.")
