🔍 Hidden Camera Detector (Real-Time + Dataset-Based)
📌 Overview

The Hidden Camera Detector is a real-time and dataset-based computer vision project designed to detect hidden cameras using a custom-trained deep learning model and live video feed analysis. The system leverages image classification techniques and real-time object detection to spot suspicious objects resembling camera lenses, making it suitable for security, privacy protection, and surveillance safety.

This project combines two detection modes:

Dataset-Based Detection – Uses a trained deep learning model on a custom dataset of real and fake cameras to classify images or frames.

Real-Time Detection – Uses the system’s webcam to continuously scan the environment for hidden cameras and display the live feed with detection results.

🎯 Features

✅ Dual Mode Detection – Dataset-trained classification + Real-time camera scanning.
✅ Live Feed Monitoring – See exactly what your camera sees while detection is happening.
✅ No Frame Count Display – Only detects and displays possible camera presence.
✅ Custom Dataset Support – Uses your own dataset for higher accuracy.
✅ Optimized Accuracy – Model trained with optimized hyperparameters for precise detection.
✅ Privacy Protection – Helps users identify suspicious devices in public or private spaces.

🛠 Technologies Used

Python 3.9+

PyTorch – Deep learning framework for model training & inference

OpenCV – For real-time video feed capture and image processing

Torchvision – Preprocessing transformations and model utilities

NumPy – Array operations and data handling

Custom Dataset – Collected and labeled images for training/validation/testing

📂 Project Structure
Hidden-Camera-Detector/
│
├── dataset/                  # Contains train/valid/test image folders
│   ├── train/
│   ├── valid/
│   └── test/
│
├── model/                    # Saved trained model file
│   └── hidden_camera_model.pth
│
├── real_time_detect.py       # Real-time hidden camera detection script
├── train_model.py            # Model training script
├── utils.py                  # Helper functions
├── requirements.txt          # Python dependencies
└── README.md                 # Project description

📊 Dataset Information

The dataset consists of two main classes:

Camera – Images containing visible or partially hidden cameras (including small surveillance devices).

Non-Camera – Images without cameras, including everyday objects to avoid false positives.

🚀 How It Works

Training Phase

The model is trained on a labeled dataset using a convolutional neural network (CNN).

The dataset is split into training, validation, and testing sets.

The model learns to differentiate between “Camera” and “Non-Camera” images.

Real-Time Detection Phase

OpenCV captures live video from your webcam.

Frames are preprocessed and passed to the trained model.

If a frame is classified as containing a camera, it’s displayed with a “Camera Detected” alert.

📥 Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/Hidden-Camera-Detector.git
cd Hidden-Camera-Detector

2️⃣ Install Dependencies
pip install -r requirements.txt

▶ Usage
Train the Model
python train_model.py

Run Real-Time Detection
python real_time_detect.py

📌 Future Improvements

Add YOLO-based object detection for bounding box localization.

Include infrared reflection detection for enhanced accuracy in low-light.

Build a mobile app version for portable scanning.

Implement multi-camera scanning for security rooms.
