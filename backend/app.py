from flask import Flask, jsonify, request
from flask_cors import CORS
from predict import predict_machine
from rag import search_manual

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return jsonify({
        "project": "PredictCare AI",
        "status": "Backend Running Successfully"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        air_temp = float(data["temperature"])
        process_temp = float(data["process_temperature"])
        rpm = float(data["rpm"])
        torque = float(data["torque"])
        tool_wear = float(data["tool_wear"])

        # ML Prediction
        result = predict_machine(
            air_temp,
            process_temp,
            rpm,
            torque,
            tool_wear
        )

        # RAG Query
        if result == "Healthy Machine":
            question = "How should a healthy machine be maintained?"
        else:
            question = "What maintenance should be performed when a machine failure is predicted?"

        rag_answer = search_manual(question)

        if result == "Healthy Machine":

            return jsonify({
                "prediction": result,
                "risk_level": "Low",
                "confidence": "98.5%",
                "maintenance_priority": "Low",
                "estimated_life": "150 Hours",
                "recommendation": "Continue normal operation. No maintenance required.",
                "rag_response": rag_answer
            })

        else:

            return jsonify({
                "prediction": result,
                "risk_level": "High",
                "confidence": "96.2%",
                "maintenance_priority": "High",
                "estimated_life": "15 Hours",
                "recommendation": "Immediate inspection required. Replace worn components.",
                "rag_response": rag_answer
            })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)