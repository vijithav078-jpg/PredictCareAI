import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("../dataset/ai4i2020.csv")

# Select features
X = df.iloc[:, 3:8]

# Target
y = df["Machine failure"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "saved_models/predictive_model.pkl")

print("Model trained successfully!")