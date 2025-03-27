from flask import Flask, request, jsonify
import pickle
import os
import base64
from train import train_models

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# === Model Loading ===

models = {}
material_mapping = {}  # base64 -> decoded name

MODEL_DIR = "models"

def load_models():
    global models, material_mapping
    models.clear()
    material_mapping.clear()

    print("🔄 Loading models from:", MODEL_DIR)

    for file in os.listdir(MODEL_DIR):
        if file.startswith("arima_") and file.endswith(".pkl"):
            encoded_part = file[len("arima_"):-len(".pkl")]
            try:
                decoded_name = base64.b64decode(encoded_part.encode()).decode()
                filepath = os.path.join(MODEL_DIR, file)
                with open(filepath, "rb") as f:
                    models[decoded_name] = pickle.load(f)
                material_mapping[decoded_name] = encoded_part
            except Exception as e:
                print(f"❌ Failed to load {file}: {e}")
                continue

    print("✅ Loaded models:")
    for key in models:
        print(f" - [{key}]")

# Load models on app start
load_models()

# === Predict Route ===

@app.route('/predict', methods=['GET'])
def predict():
    item_type = request.args.get("type")
    item_name = request.args.get("name")
    steps = int(request.args.get("steps", 1))

    print(f"📥 Predict request: type={item_type}, name='{item_name}', steps={steps}")

    if not item_name:
        return jsonify({"error": "Missing 'name' parameter"}), 400

    # Normalize input
    normalized_input = item_name.strip().lower()
    possible_keys = [
        normalized_input,
        f"{item_type}_{normalized_input}",
    ]

    matched_key = None
    for key in models:
        if key.strip().lower() in possible_keys:
            matched_key = key
            break

    if not matched_key:
        return jsonify({"error": f"No model found for '{item_name}'"}), 404

    try:
        model = models[matched_key]
        forecast = model.forecast(steps=steps)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    return jsonify({
        "type": item_type,
        "name": matched_key,
        "forecast": list(forecast)
    })

# === Train Route ===

@app.route('/train', methods=['GET'])
def train():
    connection_string = 'postgresql://postgres.viculrdtittnlgikngxg:9ouZiUP4JK6F45ST@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres'

    success, results = train_models(connection_string)

    # Reload models after training
    load_models()

    if success:
        return jsonify({
            "status": "success",
            "message": "Models trained successfully",
            "details": results
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Error in training process",
            "details": results
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
