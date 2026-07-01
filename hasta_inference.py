import cv2
import mediapipe as mp
import numpy as np
import joblib  # Used to load your trained model

# 1. Load your trained model (Save your trained Random Forest model using joblib.dump(clf, 'hasta_model.pkl'))
try:
    clf = joblib.load('rf_model.pkl')
    print("Pre-trained Hasta Model loaded successfully!")
except FileNotFoundError:
    print("Error: 'hasta_model.pkl' not found. Please train and save your model first.")
    exit()


# 2. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,         # Focusing on Asamyuta (single hand)
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def extract_live_features(hand_landmarks):
    """
    Applies the exact same normalization and feature extraction pipeline
    used during the dataset training phase.
    """
    # Extract raw (x, y, z) coordinates
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
    # --- STEP 1: SPATIAL NORMALIZATION ---
    wrist = coords[0]
    translated_coords = coords - wrist  # Move origin to wrist (0,0,0)
    
    # Scale normalization using wrist-to-middle-finger-mcp distance (Landmark 9)
    scale_factor = np.linalg.norm(coords[9] - coords[0])
    if scale_factor == 0: 
        scale_factor = 1e-6 # Avoid division by zero
    normalized_coords = translated_coords / scale_factor
    
    # --- STEP 2: GEOMETRIC FEATURE EXTRACTION ---
    features = normalized_coords.flatten().tolist()
    
    # Tip-to-tip distances (Thumb-4, Index-8, Middle-12, Ring-16, Pinky-20)
    tips = [4, 8, 12, 16, 20]
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            dist = np.linalg.norm(normalized_coords[tips[i]] - normalized_coords[tips[j]])
            features.append(dist)
            
    return np.array(features).reshape(1, -1)

# 3. Start Live Video Capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Change index to 1 or 2 if using an external Intel RealSense camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting real-time inference loop. Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a selfie-view display, then convert BGR to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame through MediaPipe
    results = hands.process(rgb_frame)
    
    current_prediction = "Detecting Hand..."
    prediction_confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw visual skeleton landmarks on the live screen
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract normalized geometry
            live_features = extract_live_features(hand_landmarks)
            
            # Run prediction and fetch probabilities
            pred_class = clf.predict(live_features)[0]
            pred_proba = clf.predict_proba(live_features)[0]
            max_proba = np.max(pred_proba)
            
            
            print("Prediction:", pred_class)
            print("--------------------------------")
            
            # Filter low confidence predictions to reduce fluttering/noise
            if max_proba > 0.40:
                current_prediction = pred_class.replace("_", " ").title()
                prediction_confidence = max_proba * 100
            else:
                current_prediction = "Transitioning..."

    # 5. Overlay Results on the UI Frame
    cv2.rectangle(frame, (10, 10), (450, 85), (0, 0, 0), -1) # Background bar
    cv2.putText(frame, f"Mudra: {current_prediction}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if prediction_confidence > 0:
        cv2.putText(frame, f"Confidence: {prediction_confidence:.1f}%", (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Display window
    cv2.imshow('Real-Time Bharatanatyam Mudra Classifier', frame)
    
    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()