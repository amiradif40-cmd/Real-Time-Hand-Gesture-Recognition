import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os 

# --- 1. Définir le "device" (CPU) ---
device = torch.device("cpu")
print(f"Test sur : {device}")

# --- 2. Recréer l'Architecture du Modèle ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 3. Charger le Modèle Entraîné et les Classes ---
MODEL_SAVE_PATH = "hand_gesture_model.pth"
CLASSES_FILE_PATH = "classes.txt" 

if not os.path.exists(MODEL_SAVE_PATH):
    print(f"ERREUR: Fichier modèle '{MODEL_SAVE_PATH}' non trouvé. Avez-vous lancé train.py ?")
    exit()
if not os.path.exists(CLASSES_FILE_PATH):
    print(f"ERREUR: Fichier 'classes.txt' non trouvé. Avez-vous lancé train.py ?")
    exit()

with open(CLASSES_FILE_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

NUM_CLASSES = len(class_names)
print(f"Classes chargées : {class_names} (Total: {NUM_CLASSES})")

# Initialiser le modèle
model = SimpleCNN(num_classes=NUM_CLASSES).to(device)

# Charger les poids  sauvegardés
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval() 

# --- 4. Définir les Transformations d'Image ---
data_transforms = transforms.Compose([
    transforms.ToPILImage(), # Convertir l'array NumPy (OpenCV) en image PIL
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --- 5. Initialiser MediaPipe Hands ---
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

padding = 20

# --- 6. Démarrer la Boucle de Test en Temps Réel ---
print("... Démarrage du test en temps réel ... Appuyez sur 'q' pour quitter.")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    H, W, _ = frame.shape
    
    prediction_text = "Pas de main"
    text_position = (20, 40) # Position (coin supérieur gauche)
    text_color = (0, 0, 255) # Couleur (Rouge)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Calculer la Bounding Box
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        x_min_px = int(x_min * W)
        x_max_px = int(x_max * W)
        y_min_px = int(y_min * H)
        y_max_px = int(y_max * H)

        x_min_px_padded = max(0, x_min_px - padding)
        y_min_px_padded = max(0, y_min_px - padding)
        x_max_px_padded = min(W, x_max_px + padding)
        y_max_px_padded = min(H, y_max_px + padding)
        
        # Dessiner la bounding box
        cv2.rectangle(frame, (x_min_px_padded, y_min_px_padded), (x_max_px_padded, y_max_px_padded), (0, 255, 0), 2)
        
        # --- 7. Préparation de l'Image pour le Modèle ---
        cropped_hand = frame[y_min_px_padded:y_max_px_padded, x_min_px_padded:x_max_px_padded]
        
        if cropped_hand.size > 0:
            cropped_hand_rgb = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2RGB)
            image_tensor = data_transforms(cropped_hand_rgb)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            # --- 8. FAIRE LA PRÉDICTION ---
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted_idx = torch.max(outputs.data, 1)
                prediction_text = class_names[predicted_idx.item()]
    
        # --- 9. Mettre à jour la position et la couleur du texte ---
        text_position = (x_min_px_padded, y_min_px_padded - 10) # Position juste au-dessus de la boîte
        text_color = (0, 255, 0) # Couleur (Vert)
    
    # --- 10. Afficher la Prédiction (MAINTENANT SANS ERREUR) ---

    cv2.putText(
        frame, 
        prediction_text.upper(), 
        text_position, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        text_color, 
        2, 
        cv2.LINE_AA
    )
    
    cv2.imshow('Test en Temps Reel', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 11. Nettoyage ---
hands.close()
cap.release()
cv2.destroyAllWindows()

