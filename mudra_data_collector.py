import cv2
import mediapipe as mp
import csv
import os

# 1. Setup MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

csv_filename = "mudra_dataset.csv"

# 3. Create the CSV header if it doesn't exist
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(42):  # 42 landmarks total (0-20 Left, 21-41 Right)
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

# 4. Start the Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened!")
    exit()

print("=========================================")
print("   BHARATANATYAM MUDRA DATA COLLECTOR")
print("=========================================")
print("Instructions:")
print("1. Enter Mudra name.")
print("2. Press and HOLD SPACEBAR to record frames.")
print("3. Press 'n' to type a new Mudra name.")
print("4. Press 'q' to quit.")
print("=========================================\n")

current_mudra = input("Enter Mudra name to start: ").strip().lower()

# Lowered confidence thresholds to 0.5 to help with overlapping hands
with mp_hands.Hands(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    max_num_hands=2) as hands: 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Display UI labels
        cv2.putText(frame, f"Label: {current_mudra.upper()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "HOLD SPACE to Record | 'n' New | 'q' Quit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Catch keypress ONCE per frame loop
        key = cv2.waitKey(1) & 0xFF
        recording = (key == 32)  # 32 is Spacebar

        # Initialize empty structures for both possible hands
        left_hand_coords = [0.0] * 63   # 21 landmarks * 3 coordinates
        right_hand_coords = [0.0] * 63
        left_detected = False
        right_detected = False

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw skeleton
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract coordinates
                coords = []
                for landmark in hand_landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                
                # Assign to correct side based on MediaPipe's classification
                hand_label = handedness.classification[0].label # "Left" or "Right"
                if hand_label == "Left":
                    left_hand_coords = coords
                    left_detected = True
                else:
                    right_hand_coords = coords
                    right_detected = True

            # If spacebar is pressed, save data
            if recording and left_detected and right_detected:
                # OPTIONAL STRATEGY: If it's a double-hand mudra but one hand drops out, don't record junk data
                # You can remove this check if you are recording single-hand (Asamyuta) mudras
                row = [current_mudra]
                row.extend(left_hand_coords + right_hand_coords) # Always Left then Right sequential order
                
                with open(csv_filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                cv2.putText(frame, "RECORDING...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow('Mudra Data Collector', frame)

        # Handle menu navigation keys
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_mudra = input("\nEnter new Mudra name: ").strip().lower()

cap.release()
cv2.destroyAllWindows()
print(f"\nData saved cleanly to {csv_filename}")