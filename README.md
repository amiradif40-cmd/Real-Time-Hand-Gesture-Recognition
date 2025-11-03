Reconnaissance de Gestes de la Main en Temps Réel (OpenCV & PyTorch)

==> Introduction

Ce projet est une application de Vision par Ordinateur (Computer Vision) en Python capable de reconnaître et de classifier des gestes de la main (Poing, Main Ouverte, Peace) en temps réel à partir d'un flux webcam.

L'ensemble du pipeline, de la collecte de données à l'inférence en direct, est implémenté à l'aide d'OpenCV pour la capture d'image, MediaPipe pour la détection de la main, et PyTorch pour l'entraînement d'un modèle de Deep Learning (CNN).

==>  Technologies Utilisées

Python 3.10+

PyTorch : Pour la création et l'entraînement du Réseau de Neurones Convolutif (CNN).

OpenCV : Pour la capture vidéo, le traitement d'image et l'affichage.

MediaPipe (par Google) : Pour la détection haute performance des points-clés (landmarks) de la main.

Numpy : Pour la manipulation des images.

==>  Structure du Projet

Real-Time-Hand-Gesture-Recognition/
│
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


==>  Installation et Utilisation

Ce projet a été développé sur Windows 10/11.

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
Le modèle (hand_gesture_model.pth) est déjà entraîné et inclus dans ce dépôt. Vous pouvez le lancer directement !

python test.py


Ouvrez votre main, faites un poing, et le modèle devrait le prédire en direct.

==>  Comment l'entraîner vous-même (Recréer le projet)

Phase 1 : Collecte de Données

Lancez le script de collecte. Placez votre main devant la caméra et appuyez sur les touches pour sauvegarder les images dans les dossiers data/.

p - Poing

o - Main Ouverte

v - Peace

q - Quitter

python collect_data.py


Phase 2 : Entraînement du Modèle

Une fois que vous avez collecté suffisamment d'images (300+ par classe est un bon début), lancez le script d'entraînement.

python train.py


Le script va :

Charger les images depuis le dossier data/.

Appliquer des transformations (redimensionner, N&B, normaliser).

Diviser en sets d'entraînement (80%) et de validation (20%).

Entraîner le SimpleCNN pendant 15 époques.

Sauvegarder le modèle final sous hand_gesture_model.pth.

Phase 3 : Test

Lancez test.py pour voir votre propre modèle en action !

python test.py


👤 Auteur

Amira DIF

Portfolio : amiradif40-cmd.github.io

LinkedIn : linkedin.com/in/amira-dif-605574191