'''
autrice : AMIRA DIF 
'''

import cv2
import mediapipe as mp
import os
import time

# --- 1. Initialisation de MediaPipe Hands ---
# Utilitaire pour dessiner les points et les connexions sur la main
mp_drawing = mp.solutions.drawing_utils
# Utilitaire pour le modèle de détection de main
mp_hands = mp.solutions.hands

# Configuration du modèle Hands
# max_num_hands=1 : Nous ne traitons qu'une seule main
# min_detection_confidence=0.7 : Seuil de confiance pour la détection
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

# --- 2. Configuration des Dossiers et Classes ---
DATA_DIR = 'data'

# Liste de nos classes (gestes)
classes = ['poing', 'main_ouverte', 'peace']

# Dictionnaire pour compter les images sauvegardées par classe
counters = {cls: 0 for cls in classes}

# --- 3. Démarrage de la Capture Vidéo ---
print("Démarrage de la capture...")
print(">>> Appuyez sur 'q' pour quitter.")
print(">>> Placez votre main et appuyez sur 'p' (poing), 'o' (ouverte), ou 'v' (peace) pour sauvegarder.")

# Ouvre la webcam (0 est généralement la webcam par défaut)
cap = cv2.VideoCapture(0)

# Marge (padding) autour de la boîte de détection (bounding box)
padding = 20

while cap.isOpened():
    # Lire une frame (image) de la webcam
    success, frame = cap.read()
    if not success:
        print("Ignorance d'une frame vide.")
        continue

    # Retourner l'image horizontalement (effet miroir)
    frame = cv2.flip(frame, 1)
    
    # Stocker les dimensions de l'image
    H, W, _ = frame.shape

    # Convertir l'image de BGR (OpenCV) à RGB (MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Traiter l'image avec MediaPipe Hands
    results = hands.process(frame_rgb)

    # Variable pour stocker l'image de la main rognée (cropped)
    cropped_hand = None
    
    # --- 4. Détection et Dessin ---
    if results.multi_hand_landmarks:
        # On ne traite que la première main détectée
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Dessiner les points (landmarks) et les connexions sur l'image
        mp_drawing.draw_landmarks(
            frame, 
            hand_landmarks, 
            mp_hands.HAND_CONNECTIONS)
        
        # Calculer la "Bounding Box" (boîte) autour de la main
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Convertir en coordonnées pixels
        x_min_px = int(x_min * W)
        x_max_px = int(x_max * W)
        y_min_px = int(y_min * H)
        y_max_px = int(y_max * H)

        # Ajouter la marge (padding) en s'assurant de ne pas sortir de l'image
        x_min_px_padded = max(0, x_min_px - padding)
        y_min_px_padded = max(0, y_min_px - padding)
        x_max_px_padded = min(W, x_max_px + padding)
        y_max_px_padded = min(H, y_max_px + padding)
        
        # Dessiner la bounding box sur l'image
        cv2.rectangle(frame, (x_min_px_padded, y_min_px_padded), (x_max_px_padded, y_max_px_padded), (0, 255, 0), 2)
        
        # Rogner (crop) l'image de la main
        cropped_hand = frame[y_min_px_padded:y_max_px_padded, x_min_px_padded:x_max_px_padded]

    # Afficher l'image (avec les dessins) dans une fenêtre
    cv2.imshow('Collecte de Donnees', frame)

    # --- 5. Gestion des Sauvegardes ---
    # Attendre une touche (1 milliseconde)
    key = cv2.waitKey(1) & 0xFF

    # Quitter si 'q' est pressé
    if key == ord('q'):
        break

    # Définir la classe à sauvegarder en fonction de la touche
    current_class_to_save = None
    if key == ord('p'):
        current_class_to_save = 'poing'
    elif key == ord('o'):
        current_class_to_save = 'main_ouverte'
    elif key == ord('v'):
        current_class_to_save = 'peace'

    # Si une touche a été pressée ET qu'une main est détectée ET que l'image rognée existe
    if current_class_to_save and cropped_hand is not None and cropped_hand.size > 0:
        # 1. Mettre à jour le compteur
        counters[current_class_to_save] += 1
        
        # 2. Définir le chemin de sauvegarde
        class_path = os.path.join(DATA_DIR, current_class_to_save)
        
        # 3. Créer le nom de fichier (ex: poing_0001.png)
        filename = f"{current_class_to_save}_{counters[current_class_to_save]:04d}.png"
        save_path = os.path.join(class_path, filename)
        
        # 4. Sauvegarder l'image rognée
        cv2.imwrite(save_path, cropped_hand)
        print(f"Sauvegardé : {save_path}")
        
        time.sleep(0.1)

# --- 6. Nettoyage ---
hands.close()
cap.release()
cv2.destroyAllWindows()

