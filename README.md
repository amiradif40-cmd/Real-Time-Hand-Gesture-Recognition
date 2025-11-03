Reconnaissance de Gestes de la Main en Temps Réel (OpenCV & PyTorch)

🚀 Démonstration

Voici le modèle en action, capable de classifier les gestes en temps réel.

| ![Geste Poing](demo/poing.jpg) | ![Geste Main Ouverte](demo/mainouverte.jpg) | ![Geste Peace](demo/peace.jpg) |
Copiez ce nouveau `README.md` sur GitHub, et vos images s'afficheront !
Poing

Main Ouverte

Peace







(Note : Ces images sont des captures d'écran du script test.py en direct.)

📌 Introduction

Cette application Python de Vision par Ordinateur classifie les gestes de la main (Poing, Main Ouverte, Peace) en temps réel via webcam. Le pipeline complet (collecte, entraînement, inférence) utilise OpenCV, MediaPipe et un CNN PyTorch.

✨ Technologies Utilisées

Python 3.10+

PyTorch : Pour la création et l'entraînement du Réseau de Neurones Convolutif (CNN).

OpenCV : Pour la capture vidéo, le traitement d'image et l'affichage.

MediaPipe (par Google) : Pour la détection haute performance des points-clés (landmarks) de la main.

Numpy : Pour la manipulation des images.

📂 Structure du Projet

Real-Time-Hand-Gesture-Recognition/
│
├── demo/                 # Contient les images de démonstration.
├── data/                 # (Ignoré par .gitignore) Données brutes de collecte.
├── .venv/                # (Ignoré par .gitignore) Environnement virtuel.
│
├── collect_data.py       # PHASE 1: Script pour collecter les images d'entraînement.
├── train.py              # PHASE 2: Script pour entraîner le modèle CNN avec PyTorch.
├── test.py               # PHASE 3: Script pour tester le modèle en temps réel.
│
├── hand_gesture_model.pth  # Le "cerveau" IA entraîné (modèle sauvegardé).
├── classes.txt           # Fichier listant les classes (gestes) entraînées.
├── requirements.txt      # Liste des bibliothèques Python nécessaires.
├── .gitignore            # Fichier pour ignorer les dossiers non nécessaires.
└── README.md             # Vous êtes ici !



🚀 Installation et Utilisation

Le modèle (hand_gesture_model.pth) est pré-entraîné et inclus. Vous pouvez le lancer directement.

1. Cloner le Dépôt

git clone [https://github.com/amiradif40-cmd/Real-Time-Hand-Gesture-Recognition.git](https://github.com/amiradif40-cmd/Real-Time-Hand-Gesture-Recognition.git)
cd Real-Time-Hand-Gesture-Recognition



2. Créer et Activer un Environnement Virtuel

# Créer l'environnement
python -m venv .venv

# Activer (Windows)
.\.venv\Scripts\activate.bat



3. Installer les Dépendances
(La version CPU de PyTorch est spécifiée pour une compatibilité maximale)

pip install -r requirements.txt



4. Tester en Temps Réel !

python test.py



Ouvrez votre main, faites un poing, et le modèle devrait le prédire en direct.

🛠️ Comment l'entraîner vous-même (Recréer le projet)

Phase 1 : Collecte de Données

Lancez collect_data.py. Appuyez sur les touches (p, o, v) pour sauvegarder les images de votre main.

p - Poing

o - Main Ouverte

v - Peace

q - Quitter

python collect_data.py



Phase 2 : Entraînement du Modèle

Lancez python train.py pour entraîner le modèle sur les images collectées. Le script entraînera le CNN et sauvegardera le modèle final sous hand_gesture_model.pth.

python train.py



Phase 3 : Test

Lancez test.py pour tester votre nouveau modèle.

python test.py



👤 Auteur

Amira DIF - Ingénieure Traitement Signal & Image | Machine Learning

Portfolio : amiradif40-cmd.github.io

LinkedIn : linkedin.com/in/amira-dif-605574191
