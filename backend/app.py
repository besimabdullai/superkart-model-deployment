import io
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart")
model = joblib.load("superkart_model.joblib")

REQUIRED_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

@superkart_api.get("/")
def home():
    return jsonify(
        {
            "service": "SuperKart Sales Forecasting API",
            "status": "healthy",
            "endpoints": ["/v1/predict", "/v1/predictbatch"],
        }
    )

@superkart_api.get("/health")
def health():
    return jsonify({"status": "ok"})

@superkart_api.post("/v1/predict")
def predict_sales():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be valid JSON."}), 400

        missing = [feature for feature in REQUIRED_FEATURES if feature not in payload]
        if missing:
            return jsonify({"error": f"Missing required features: {missing}"}), 400

        sample = {feature: payload[feature] for feature in REQUIRED_FEATURES}
        input_data = pd.DataFrame([sample])

        prediction = float(model.predict(input_data)[0])
        if not np.isfinite(prediction):
            return jsonify({"error": "Model produced a non-finite prediction."}), 500

        return jsonify({"Sales": round(prediction, 2)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Upload a CSV file using form field 'file'."}), 400

        input_data = pd.read_csv(request.files["file"])
        missing = [feature for feature in REQUIRED_FEATURES if feature not in input_data.columns]
        if missing:
            return jsonify({"error": f"Missing required columns: {missing}"}), 400

        predictions = model.predict(input_data[REQUIRED_FEATURES])
        output = {
            str(i): round(float(pred), 2)
            for i, pred in enumerate(predictions)
        }
        return jsonify(output)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
