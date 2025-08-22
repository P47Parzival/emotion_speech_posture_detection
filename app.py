# # app.py
# import cv2
# import numpy as np
# from flask import Flask, render_template, Response
# from skimage import feature
# from skimage.feature import hog
# import joblib

# # --- 1. INITIALIZATION ---
# app = Flask(__name__)

# # Emotion model config
# try:
#     model = joblib.load('./Emotion detection/ensemble_model_fitted.pkl')
#     scaler = joblib.load('./Emotion detection/scaler_new.pkl')
#     pca = joblib.load('./Emotion detection/pca_new.pkl')
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     print("✅ Model, scaler, PCA, and cascade classifier loaded successfully!")
# except Exception as e:
#     print(f"❌ Error loading files: {e}")
#     exit()

# # Posture model config
# try:
#     posture_model = joblib.load('./Posture detection/posture_model.pkl')
#     # Create a background subtractor object
#     bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
#     print("✅ Posture detection files loaded successfully!")
# except Exception as e:
#     print(f"❌ Error loading posture files: {e}")
#     # exit

# camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# if not camera.isOpened():
#     print("❌ Cannot open camera")
#     exit()

# # Emotion, Posture labels mapping
# emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# posture_labels = {0: 'Bending', 1: 'Lying', 2: 'Sitting', 3: 'Standing'}

# # --- 2. FEATURE EXTRACTION FUNCTION (must be identical to training) ---
# def get_lbp_features(image):
#     lbp = feature.local_binary_pattern(image, P=24, R=8, method='uniform')
#     (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
#     hist = hist.astype("float")
#     hist /= (hist.sum() + 1e-6)
#     return hist

# def get_hog_features(image):
#     features = hog(image, orientations=9, pixels_per_cell=(8, 8),
#                    cells_per_block=(2, 2), block_norm="L2-Hys", visualize=False)
#     return features

# def extract_hu_moments_from_mask(mask):
#     # (Same as in Colab, but takes a mask as input)
#     moments = cv2.moments(mask)
#     hu_moments = cv2.HuMoments(moments)
#     hu_moments = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-7)
#     return hu_moments.flatten()

# # --- 3. VIDEO STREAMING GENERATOR ---
# def generate_frames():
#     # camera = cv2.VideoCapture(0)
#     global camera
#     if not camera.isOpened():
#         print("❌ Cannot open camera")
#         return

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
        
#         # Convert frame to grayscale for face detection and LBP
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Detect faces
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
#         # Process each detected face
#         for (x, y, w, h) in faces:
#             # Draw a rectangle around the face
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
#             # Crop the face region (Region of Interest - ROI)
#             roi_gray = gray[y:y+h, x:x+w]
            
#             # Resize to 48x48 (the size our model expects)
#             roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
#             # --- Extract LBP features ---
#             lbp_features = get_lbp_features(roi_resized)

#             # --- Extract HOG features ---
#             hog_features = get_hog_features(roi_resized)

#             # --- Combine LBP + HOG ---
#             combined_features = np.hstack([lbp_features, hog_features])

#             # --- Scale features ---
#             scaled_features = scaler.transform(combined_features.reshape(1, -1))

#             # --- Apply PCA (must match training) ---
#             pca_features = pca.transform(scaled_features)

#             # --- Predict emotion ---
#             prediction = model.predict(pca_features)
#             predicted_emotion = emotion_labels[prediction[0]]
            
#             # Create feedback text
#             feedback_text = f"Emotion: {predicted_emotion}"
#             if predicted_emotion in ['Sad', 'Angry', 'Fear']:
#                 feedback_text += " - Try to smile for a confident look!"
#             elif predicted_emotion == 'Happy':
#                 feedback_text += " - Great smile!"

#             # Put the feedback text on the frame
#             cv2.putText(frame, feedback_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         # Encode the frame in JPEG format
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
        
#         # Yield the frame for streaming
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     camera.release()

# # --- Posture Feed Generator ---
# def generate_posture_frames():
#     # camera = cv2.VideoCapture(0) # Use the same camera
#     global camera

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         # Apply background subtraction to get the foreground mask
#         fg_mask = bg_subtractor.apply(frame)

#         # Clean up the mask using morphological operations to remove noise
#         kernel = np.ones((5, 5), np.uint8)
#         fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
#         fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

#         # Find contours in the mask
#         contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         if contours:
#             # Find the largest contour (presumably the user)
#             largest_contour = max(contours, key=cv2.contourArea)
            
#             if cv2.contourArea(largest_contour) > 5000: # Threshold to avoid small noise
#                 # Draw the contour on the original frame
#                 cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
                
#                 # Create a mask for Hu Moments calculation
#                 contour_mask = np.zeros(frame.shape[:2], dtype="uint8")
#                 cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)

#                 # Extract features and predict
#                 hu_features = extract_hu_moments_from_mask(contour_mask)
#                 prediction = posture_model.predict(hu_features.reshape(1, -1))
#                 predicted_posture = posture_labels[prediction[0]]

#                 # --- Rule-based feedback for sitting posture ---
#                 feedback_text = f"Posture: {predicted_posture}"
#                 if predicted_posture == 'Sitting':
#                     x, y, w, h = cv2.boundingRect(largest_contour)
#                     aspect_ratio = h / w
#                     if aspect_ratio < 1.5:
#                         feedback_text += " - Sit up straight!"
#                     else:
#                         feedback_text += " - Good posture!"

#                 cv2.putText(frame, feedback_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#     camera.release()

# # --- 4. FLASK ROUTES ---
# @app.route('/')
# def index():
#     """Video streaming home page."""
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route. Put this in the src attribute of an img tag."""
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/posture_feed')
# def posture_feed():
#     return Response(generate_posture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # --- 5. MAIN EXECUTION ---
# if __name__ == '__main__':
#     app.run(debug=True)

import cv2
import numpy as np
from flask import Flask, render_template, Response
from skimage import feature
from skimage.feature import hog
import joblib
import time
import threading

# --- 1. INITIALIZATION ---
app = Flask(__name__)

# Emotion model config
try:
    model = joblib.load('./Emotion detection/ensemble_model_fitted.pkl')
    scaler = joblib.load('./Emotion detection/scaler_new.pkl')
    pca = joblib.load('./Emotion detection/pca_new.pkl')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("❌ Error: haarcascade_frontalface_default.xml not loaded correctly")
        exit()
    print("✅ Model, scaler, PCA, and cascade classifier loaded successfully!")
except Exception as e:
    print(f"❌ Error loading emotion files: {e}")
    exit()

# Posture model config
try:
    posture_model = joblib.load('./Posture detection/posture_model.pkl')
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    print("✅ Posture detection files loaded successfully!")
except Exception as e:
    print(f"❌ Error loading posture files: {e}")
    exit()

# Single shared camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    print("❌ Cannot open camera")
    exit()
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)
print("✅ Shared camera initialized")

# Global frame buffer and lock for thread safety
latest_frame = None
frame_lock = threading.Lock()

# Background thread to read frames
def read_frames():
    global latest_frame
    while True:
        success, frame = camera.read()
        if success and frame is not None:
            with frame_lock:
                latest_frame = frame.copy()
        else:
            print("❌ Failed to read frame in background thread")
            time.sleep(0.1)

# Start background thread
threading.Thread(target=read_frames, daemon=True).start()

# Emotion, Posture labels mapping
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
posture_labels = {0: 'Bending', 1: 'Lying', 2: 'Sitting', 3: 'Standing'}

# --- 2. FEATURE EXTRACTION FUNCTIONS ---
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

def extract_hu_moments_from_mask(mask):
    moments = cv2.moments(mask)
    hu_moments = cv2.HuMoments(moments)
    hu_moments = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-7)
    return hu_moments.flatten()

# --- 3. VIDEO STREAMING GENERATOR ---
def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                print("❌ No frame available for emotion feed")
                time.sleep(0.1)
                continue
            frame = latest_frame.copy()
        
        # Convert frame to grayscale for face detection and LBP
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Process each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            lbp_features = get_lbp_features(roi_resized)
            hog_features = get_hog_features(roi_resized)
            combined_features = np.hstack([lbp_features, hog_features])
            scaled_features = scaler.transform(combined_features.reshape(1, -1))
            pca_features = pca.transform(scaled_features)
            prediction = model.predict(pca_features)
            predicted_emotion = emotion_labels[prediction[0]]
            feedback_text = f"Emotion: {predicted_emotion}"
            if predicted_emotion in ['Sad', 'Angry', 'Fear']:
                feedback_text += " - Try to smile for a confident look!"
            elif predicted_emotion == 'Happy':
                feedback_text += " - Great smile!"
            cv2.putText(frame, feedback_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret:
            print("❌ Failed to encode frame for emotion feed")
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Posture Feed Generator ---
def generate_posture_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                print("❌ No frame available for posture feed")
                time.sleep(0.1)
                continue
            frame = latest_frame.copy()

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 5000:
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
                contour_mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
                hu_features = extract_hu_moments_from_mask(contour_mask)
                prediction = posture_model.predict(hu_features.reshape(1, -1))
                predicted_posture = posture_labels[prediction[0]]
                feedback_text = f"Posture: {predicted_posture}"
                if predicted_posture == 'Sitting':
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    aspect_ratio = h / w
                    if aspect_ratio < 1.5:
                        feedback_text += " - Sit up straight!"
                    else:
                        feedback_text += " - Good posture!"
                cv2.putText(frame, feedback_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret:
            print("❌ Failed to encode frame for posture feed")
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- 4. FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/posture_feed')
def posture_feed():
    return Response(generate_posture_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Flask app error: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("✅ Shared camera released")