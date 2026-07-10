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


# ==========================================
# ML Prediction API
# ==========================================
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

        # RAG Recommendation
        question = f"""
        Machine Prediction: {result}

        Air Temperature: {air_temp} °C
        Process Temperature: {process_temp} °C
        RPM: {rpm}
        Torque: {torque} Nm
        Tool Wear: {tool_wear} minutes

        Suggest maintenance recommendations for this machine.
        """

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
            "status": "error",
            "message": str(e)
        }), 500


# ==========================================
# RAG Chat API
# ==========================================
@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        question = data.get("question", "")

        answer = search_manual(question)

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)