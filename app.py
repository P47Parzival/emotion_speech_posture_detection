# app.py
import cv2
import numpy as np
from flask import Flask, render_template, Response
from skimage import feature
from skimage.feature import hog
import joblib

# --- 1. INITIALIZATION ---
app = Flask(__name__)

# Load the trained model, scaler, and face detector
try:
    model = joblib.load('./Emotion detection/ensemble_model_fitted.pkl')
    scaler = joblib.load('./Emotion detection/scaler_new.pkl')
    pca = joblib.load('./Emotion detection/pca_new.pkl')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("✅ Model, scaler, PCA, and cascade classifier loaded successfully!")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit()

# Emotion labels mapping
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# --- 2. FEATURE EXTRACTION FUNCTION (must be identical to training) ---
def get_lbp_features(image):
    lbp = feature.local_binary_pattern(image, P=24, R=8, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def get_hog_features(image):
    features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm="L2-Hys", visualize=False)
    return features

# --- 3. VIDEO STREAMING GENERATOR ---
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Convert frame to grayscale for face detection and LBP
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Crop the face region (Region of Interest - ROI)
            roi_gray = gray[y:y+h, x:x+w]
            
            # Resize to 48x48 (the size our model expects)
            roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            # --- Extract LBP features ---
            lbp_features = get_lbp_features(roi_resized)

            # --- Extract HOG features ---
            hog_features = get_hog_features(roi_resized)

            # --- Combine LBP + HOG ---
            combined_features = np.hstack([lbp_features, hog_features])

            # --- Scale features ---
            scaled_features = scaler.transform(combined_features.reshape(1, -1))

            # --- Apply PCA (must match training) ---
            pca_features = pca.transform(scaled_features)

            # --- Predict emotion ---
            prediction = model.predict(pca_features)
            predicted_emotion = emotion_labels[prediction[0]]
            
            # Create feedback text
            feedback_text = f"Emotion: {predicted_emotion}"
            if predicted_emotion in ['Sad', 'Angry', 'Fear']:
                feedback_text += " - Try to smile for a confident look!"
            elif predicted_emotion == 'Happy':
                feedback_text += " - Great smile!"

            # Put the feedback text on the frame
            cv2.putText(frame, feedback_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# --- 4. FLASK ROUTES ---
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    app.run(debug=True)