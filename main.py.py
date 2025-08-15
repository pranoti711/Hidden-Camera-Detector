import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_curve
import cv2
from PIL import Image
import keyboard  # NEW for global key detection

# === Paths ===
TRAIN_DIR = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\hidden camera detection\train"
VALID_DIR = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\hidden camera detection\valid"
TEST_DIR  = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\hidden camera detection\test"
MODEL_PATH = "hybrid_hidden_camera_model.pt"

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Data Loaders ===
train_loader = DataLoader(ImageFolder(TRAIN_DIR, transform=transform), batch_size=16, shuffle=True)
valid_loader = DataLoader(ImageFolder(VALID_DIR, transform=transform), batch_size=16)
test_loader  = DataLoader(ImageFolder(TEST_DIR,  transform=transform), batch_size=16)

# === Model Setup ===
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === Training ===
def train_model(epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

        train_acc = 100 * correct / total
        print(f"\U0001F4D8 Epoch {epoch+1}/{epochs} | Train Accuracy: {train_acc:.2f}% | Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)

# === Threshold Optimization ===
def optimize_threshold():
    model.eval()
    y_true, y_scores = [], []
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            outputs = model(images)
            scores = torch.sigmoid(outputs).cpu().numpy().flatten()
            y_scores.extend(scores)
            y_true.extend(labels.numpy())
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_thresh = thresholds[np.argmax(f1)]
    print(f"\U0001F527 Optimized Threshold from Validation Set: {best_thresh:.2f}")
    return best_thresh

# === Test Accuracy ===
def test_accuracy(threshold):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = torch.sigmoid(model(images))
            preds = (outputs > threshold).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"\U0001F9EA Test Accuracy: {acc:.2f}%")
    return acc

# === Real-Time Detection ===
def real_time_detection(threshold):
    user_confirm = input("\U0001F440 Allow system camera access to display live view and detect hidden cameras? (y/n): ").strip().lower()
    if user_confirm != 'y':
        print("\u274C Camera access denied by user.")
        return

    model.eval()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\u26A0\ufe0f Unable to access system camera.")
        return

    print("\U0001F3A5 Real-time detection started. Press ESC or type 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\u274C Failed to read frame from camera.")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img).resize((224, 224))
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            score = torch.sigmoid(model(img_tensor)).item()

        label = "Camera Detected!" if score > threshold else "No Camera"
        color = (0, 0, 255) if score > threshold else (0, 255, 0)
        cv2.putText(frame, f"{label} ({score:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Live Hidden Camera Detection View", frame)

        # Detect global key press
        if keyboard.is_pressed('esc') or keyboard.is_pressed('q'):
            print("\n⏹️ ESC or 'q' detected. Exiting real-time detection.")
            break

        if cv2.getWindowProperty("Live Hidden Camera Detection View", cv2.WND_PROP_VISIBLE) < 1:
            print("\n📴 Window closed. Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\U0001F3C1 Hybrid Hidden Camera Detection Completed.")

# === Main Execution ===
if __name__ == "__main__":
    print("\U0001F4CC Hybrid Hidden Camera Detection Started...")

    if not os.path.exists(MODEL_PATH):
        print("\U0001F6E0️ Training model for 5 epochs...")
        train_model(epochs=5)
    else:
        print("\U0001F4C5 Model found. Loading...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    best_thresh = optimize_threshold()
    print(f"\U0001F527 Using optimized threshold: {best_thresh:.2f}")

    test_accuracy(best_thresh)
    real_time_detection(best_thresh)
