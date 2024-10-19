import cv2
import os
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize Mediapipe Hand Landmarker
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to extract hand landmarks
def extract_hand_landmarks(image, hands):
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

# Function to preprocess all images from the raw data folder and save to individual CSV files per label
def preprocess_data_separate_csv(raw_data_path, output_csv_path):
    # Initialize Mediapipe hands detection
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        
        # Loop through each folder (label) in the raw data directory
        for label in os.listdir(raw_data_path):
            label_path = os.path.join(raw_data_path, label)
            if os.path.isdir(label_path):
                data = []

                # Loop through each image file in the label folder
                for image_file in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_file)
                    image = cv2.imread(image_path)
                    
                    if image is not None:
                        # Extract hand landmarks from the image
                        landmarks = extract_hand_landmarks(image, hands)
                        
                        if landmarks:
                            # Normalize the landmarks
                            normalized_landmarks = normalize_landmarks(landmarks, image.shape)
                            data.append(normalized_landmarks)
                        else:
                            print(f"No hand detected in image: {image_file}")
                
                # Save the extracted landmarks for this label to a CSV file
                if data:
                    label_csv_path = os.path.join(output_csv_path, f"{label}.csv")
                    df = pd.DataFrame(data)
                    df.to_csv(label_csv_path, index=False)
                    print(f"Data for label '{label}' saved to {label_csv_path}")
                else:
                    print(f"No valid data for label '{label}'.")

# Main function to preprocess the data and save separate CSVs per label
if __name__ == "__main__":
    raw_data_path = os.path.join("data", "raw")
    output_csv_path = os.path.join("data", "processed")

    # Create the processed folder if it doesn't exist
    if not os.path.exists(output_csv_path):
        os.makedirs(output_csv_path)

    # Start preprocessing and saving separate CSVs
    preprocess_data_separate_csv(raw_data_path, output_csv_path)

    print("Preprocessing complete!")
