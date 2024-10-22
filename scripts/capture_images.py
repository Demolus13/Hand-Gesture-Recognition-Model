import cv2
import os

import mediapipe as mp

# Capture images using the webcam
def capture_images(label, save_path, max_images=50):
    # Create a directory to store the images for the given label
    label_dir = os.path.join(save_path, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    # Start video capture
    cap = cv2.VideoCapture(0)

    print(f"Press 'Enter' to capture image for label '{label}', and 'q' to quit.")

    image_count = 0
    while image_count < max_images:
        # Read frame from webcam
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture image. Please check your camera.")
            break

        # Convert frame to RGB (required by MediaPipe)
        frame_show = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        # If hand landmarks are detected, draw them on the frame
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_show, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        # Display the frame in a window
        cv2.imshow("Capture Image - Press Enter to save, q to quit", frame_show)

        # Wait for key press (13 is Enter key, 'q' is to quit)
        key = cv2.waitKey(1)
        if key == 13:  # Enter key
            image_path = os.path.join(label_dir, f"{label}_{image_count}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Captured and saved image: {image_path}")
            image_count += 1
        elif key == ord('q'):
            print("Exiting...")
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# Main function to capture labeled images
if __name__ == "__main__":
    # Define path to store the raw images
    save_path = os.path.join("data", "raw")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Get label name from user
    label = input("Enter the label name for the images(e.g., 'THUMBS_UP', 'WAVE'): ").strip()

    # Start capturing images
    capture_images(label, save_path, max_images=50)

    print("Image capture complete!")
