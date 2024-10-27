# Hand Gesture Recognition

This repository contains code for capturing hand gesture images, preprocessing the data, training a hand gesture recognition model, and evaluating the model.

## Project Structure
```
Hand-Gesture-Recognition-Model/
│
├── data/
│   ├── processed/
│   │   ├── PAPER.csv
│   │   ├── ROCK.csv
│   │   └── SCISSORS.csv
│   └── raw/
│       ├── PAPER/
│       ├── ROCK/
│       └── SCISSORS/
│
├── models/
│   ├── hgr_model.pth
│   └── label_encodings.pkl
│
├── scripts/
│   ├── __pycache__/
│   ├── capture_images.py
│   ├── evaluate_model.py
│   ├── preprocess_data.py
│   └── train_model.py
│
├── README.md
└── requirements.txt
```

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/Demolus13/Hand-Gesture-Recognition-Model.git
    cd hand-gesture-recognition
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Capturing Images

To capture images for a specific hand gesture label (e.g., "PAPER"), run the following command:

```sh
python capture_images.py
```
It will prompt the user to enter the label of the data that is being collected, which will be stored in [`data/raw`](./data/raw/) .

This will open a webcam feed. Press 'Enter' to capture an image and 'q' to quit.

## Preprocessing Data
To preprocess the captured images and save the extracted landmarks to CSV files, run:

```sh
python preprocess_data.py
```

This will generate CSV files in the [`data/processed`](./data/processed/) directory.

## Training the Model
To train the hand gesture recognition model, run:

```sh
python train_model.py
```

This will save the trained model and label encodings in the [`models`](./models/) directory.

## Evaluating the Model
To evaluate the trained model using a webcam feed, run:

```sh
python evaluate_model.py
```

This will open a webcam feed and display the predicted hand gesture on the screen.

## File Descriptions
- `scripts/capture_images.py`: Script to capture hand gesture images using a webcam.
- `scripts/preprocess_data.py`: Script to preprocess captured images and save landmarks to CSV files.
- `scripts/train_model.py`: Script to train the hand gesture recognition model.
- `scripts/evaluate_model.py`: Script to evaluate the trained model using a webcam feed.
- `scripts/model.py`: Contains the definition of the hand gesture recognition model.

## Requirements
- Python 3.6+
- OpenCV
- Mediapipe
- PyTorch
- Pandas
- NumPy