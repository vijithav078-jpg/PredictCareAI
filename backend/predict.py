import joblib
import pandas as pd

# Load trained model
model = joblib.load("saved_models/predictive_model.pkl")

def predict_machine(air_temp, process_temp, rpm, torque, tool_wear):

    # Create dataframe
    input_data = pd.DataFrame([[
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear
    ]], columns=[
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        return "Machine Failure"
    else:
        return "Healthy Machine"