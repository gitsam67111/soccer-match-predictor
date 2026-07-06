import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from features import add_previous_5_match_points, add_elo_ratings

# Load data
df = pd.read_csv("data/raw/E0.csv")

# Create features
df = add_previous_5_match_points(df)
df = add_elo_ratings(df)

# Select features
features = [
    "home_form_points",
    "away_form_points",
    "home_elo",
    "away_elo",
]

X = df[features]
y = df["FTR"]

# Chronological split (first 80% train, last 20% test)
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2%}")
print(classification_report(y_test, predictions))

# Feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_,
})

print(importance.sort_values("Importance", ascending=False))

# Save model
MODEL_PATH = Path(__file__).parent / "model.pkl"
joblib.dump(model, MODEL_PATH)

print("Model saved to:", MODEL_PATH)
