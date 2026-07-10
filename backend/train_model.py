import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("../dataset/ai4i2020.csv")

# =====================================
# Features
# =====================================

X = df[[
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]]

# Target

y = df["Machine failure"]

# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# Random Forest
# =====================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =====================================
# Accuracy
# =====================================

pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# =====================================
# Save Model
# =====================================

joblib.dump(model, "saved_models/predictive_model.pkl")

print("\n✅ New Predictive Model Saved Successfully!")