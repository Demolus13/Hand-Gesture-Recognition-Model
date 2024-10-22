import cv2
import torch
import numpy as np

import pickle
from model import HGRModel
import mediapipe as mp

# Load the label encodings
with open('models/label_encodings.pkl', 'rb') as f:
    label_to_index = pickle.load(f)
index_to_label = {v: k for k, v in label_to_index.items()}

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HGRModel(in_features=21*2, out_features=len(label_to_index))
model.load_state_dict(torch.load('models/hgr_model.pth', map_location=device))
model.to(device)
model.eval()

# Initialize Mediapipe Hand Landmarker
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to extract hand landmarks
def extract_landmarks(frame):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)
        
        # Return the normalized landmarks if found
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            return hand_landmarks.landmark
    return None

# Function to normalize hand landmarks (coordinates between 0 and 1)
def normalize_landmarks(landmarks, image_shape):
    normalized_landmarks = []
    image_height, image_width, _ = image_shape
    for landmark in landmarks:
        normalized_landmarks.append([
            min(int(landmark.x * image_width), image_width - 1),
            min(int(landmark.y * image_height), image_height - 1)
        ])

    # Convert to NumPy array for vectorized operations
    normalized_landmarks = np.array(normalized_landmarks, dtype=np.float32)
    
    # Normalize landmarks
    normalized_landmarks = normalized_landmarks - normalized_landmarks[0]
    normalized_landmarks = normalized_landmarks / np.max(np.abs(normalized_landmarks))
    
    return normalized_landmarks.flatten()

# Function to preprocess the frame
def preprocess_frame(frame):
    # Extract landmarks from the frame
    landmarks = extract_landmarks(frame)
    if landmarks is None:
        return None
    # Normalize the landmarks
    normalized_landmarks = normalize_landmarks(landmarks, frame.shape)
    return torch.tensor(normalized_landmarks, dtype=torch.float32).to(device).unsqueeze(0)

# Open a connection to the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_tensor = preprocess_frame(frame)
    if input_tensor is None:
        continue

    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        prediction_index = predicted.item()
        prediction_label = index_to_label[prediction_index]

    # Display the prediction on the frame
    cv2.putText(frame, f'Prediction: {prediction_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()