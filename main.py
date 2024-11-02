import cv2
import mediapipe as mp
import numpy as np
import time

from model import load_model, predict, NN

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

labels = [
    'Closed Fist',
    'Finger Circle',
    'Finger Symbols',
    'Multi Finger Bend',
    'Open Palm',
    'Semi Open Fist',
    'Semi Open Palm',
    'Single Finger Bend'
]

def main():
    model = load_model()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise ValueError("Camera is not accesible.")    
    
    last_prediction_time = -1
    last_prediction = ''
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Predict current frame
        current_time = time.time()
        if current_time - last_prediction_time > 1:
            prediction = predict( frame, model )

            last_prediction = labels[prediction[0]]
            last_prediction_time = current_time

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
                y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                margin = 20
                
                x_min = max(x_min - margin, 0)
                y_min = max(y_min - margin, 0)
                x_max = min(x_max + margin, w)
                y_max = min(y_max + margin, h)
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (50, 255, 150), 2)

                (text_w, text_h), _ = cv2.getTextSize(last_prediction, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.putText(frame, last_prediction, (x_min - text_w//2 + (x_max - x_min)//2, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 150), 2)

        cv2.putText(frame, f"FPS: {cap.get(cv2.CAP_PROP_FPS)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("RNN", frame)
        
        key = cv2.waitKey(1) & 0xFF
    
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()