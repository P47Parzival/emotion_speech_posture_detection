# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # ADD THIS
import io, cv2, numpy as np
import joblib, dlib
from PIL import Image
from src.features.landmarks import LandmarkExtractor
from src.features.geometry import compute_geo28_features
from src.utils.labels import EMOTION_ORDER, idx_to_label
import os

# --- config ---
PREDICTOR_PATH = os.environ.get("PREDICTOR_PATH", "shape_predictor_68_face_landmarks.dat")
MODEL_PATH     = os.environ.get("MODEL_PATH", "artifacts/model_combo164_logreg_l2.pkl")
UPSAMPLE       = int(os.environ.get("UPSAMPLE", "1"))
FEATURES       = os.environ.get("FEATURES", "combo164")  # landmarks68 | geo28 | combo164
DROP_JAW       = os.environ.get("DROP_JAW", "false").lower() == "true"
# --------------

app = Flask(__name__)
CORS(app)  # ADD THIS - enables CORS for all routes

# load model + dlib
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(r"C:\Users\dhruv\Emotion, Posture, and Speech recognition\fer_replication_starter\fer_repl\src\shape_predictor_68_face_landmarks.dat")
LEX = LandmarkExtractor(face_detector, landmark_predictor, upsample=UPSAMPLE)

model = joblib.load(MODEL_PATH)  # trained on your dataset with same FEATURES

# Get actual classes from the model (handles case where not all 8 emotions are present)
if hasattr(model, 'classes_'):
    # For sklearn classifiers, classes_ contains the actual class indices used during training
    model_classes = model.classes_
    classes = [idx_to_label(int(i)) for i in model_classes]
else:
    # Fallback to EMOTION_ORDER if model doesn't have classes_ attribute
    classes = EMOTION_ORDER

print(f"Loaded model with {len(classes)} classes: {classes}")

def zero_out_jaw(pts68):
    pts = pts68.copy()
    pts[0:17,:] = 0.0
    return pts

def flatten_landmarks(pts68):  # 68x2 -> 136
    return pts68.reshape(-1)

def extract_features_from_bgr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    pts = LEX.detect_landmarks(gray)
    if pts is None:
        return None

    if DROP_JAW:
        pts = zero_out_jaw(pts)

    lm136 = flatten_landmarks(pts)
    geo28 = compute_geo28_features(pts)

    if FEATURES == "landmarks68":
        feats = lm136
    elif FEATURES == "geo28":
        feats = geo28
    else:
        feats = np.concatenate([lm136, geo28], axis=0)  # combo164
    return feats

@app.route("/predict", methods=["POST"])
def predict():
    if "frame" not in request.files:
        return jsonify({"error":"no frame"}), 400
    file = request.files["frame"].read()
    img = Image.open(io.BytesIO(file)).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    feats = extract_features_from_bgr(bgr)
    if feats is None:
        return jsonify({"label":"No face", "probs":{}})

    X = feats.reshape(1, -1)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[0]
        y_idx = int(np.argmax(y_prob))
    else:
        y_idx = int(model.predict(X)[0])
        # synthesize probs-like dict
        y_prob = np.zeros(len(classes)); y_prob[y_idx] = 1.0

    label = classes[y_idx]
    # Only iterate over the actual probabilities returned by the model
    probs = {classes[i]: float(y_prob[i]) for i in range(len(y_prob))}
    return jsonify({"label": label, "probs": probs})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)