'''
autrice : AMIRA DIF 
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

print(f"Version de PyTorch utilisée : {torch.__version__}")

# --- 1. Définir le "device" (GPU ou CPU) ---
#  vérifie si un GPU (NVIDIA) est disponible, sinon on utilise le CPU
# L'entraînement sera BEAUCOUP plus rapide sur un GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entraînement sur : {device}")

# --- 2. Préparation des Données (Transforms & Datasets) ---

# Définir les transformations à appliquer à chaque image
# C'est important pour la "data augmentation" et de normalisation
data_transforms = transforms.Compose([
    # Redimensionner toutes les images à la même taille (64x64 pixels)
    transforms.Resize((64, 64)),
    
    # Mettre en noir et blanc (1 seul canal). La couleur n'est pas utile ici.
    transforms.Grayscale(num_output_channels=1),

    
    # Convertir l'image en Tenseur PyTorch (format que le réseau comprend)
    transforms.ToTensor(),
    
    # Normaliser les pixels (aide le réseau à apprendre plus vite)
    # Moyenne de 0.5, écart-type de 0.5. Simple et efficace pour le N&B.
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Chemin vers notre dossier de données
DATA_DIR = 'data'

# Charger TOUTES les images en mémoire en utilisant ImageFolder
# ImageFolder est magique : il utilise automatiquement les noms de dossiers
# ('poing', 'main_ouverte', 'peace') comme "labels" (étiquettes).
full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)

# Afficher les classes trouvées (pour vérifier)
class_names = full_dataset.classes
print(f"Classes trouvées : {class_names}")

# Sauvegarder les classes dans un fichier pour le script de test
# C'est important pour que le script de test sache que 0 = 'main_ouverte', 1 = 'peace', etc.
# (L'ordre est alphabétique par défaut)
with open('classes.txt', 'w') as f:
    for item in class_names:
        f.write(f"{item}\n")

# --- 3. Séparation des Données (Train / Validation) ---
# 80% pour l'entraînement (le modèle va les "voir" et apprendre)
# 20% pour la validation (le modèle ne les "verra pas", on les garde pour le tester)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Taille du set d'entraînement : {len(train_dataset)} images")
print(f"Taille du set de validation : {len(val_dataset)} images")

# --- 4. Création des DataLoaders ---
# Les DataLoaders préparent les données en "lots" (batchs) 
# pour les donner au réseau petit à petit.
BATCH_SIZE = 16 # On traite 16 images à la fois

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 5. Définition du Modèle (Le "Cerveau" CNN) ---
# C'est l'architecture de notre Réseau de Neurones Convolutif (CNN)
# C'est un modèle simple mais très efficace pour la classification d'images.

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Couche 1 : Convolution (Entrée: 1 canal N&B, Sortie: 16 canaux)
        # (1, 64, 64) -> (16, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Couche 2 : Convolution (Entrée: 16 canaux, Sortie: 32 canaux)
        # (16, 32, 32) -> (32, 16, 16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Couche 3 : Convolution (Entrée: 32 canaux, Sortie: 64 canaux)
        # (32, 16, 16) -> (64, 8, 8)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # "Aplatir" l'image pour la couche de décision
        # 64 canaux * 8 * 8 (taille de l'image après 3 poolings)
        self.flatten = nn.Flatten()
        
        # Couche de décision (Fully Connected)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu4 = nn.ReLU()
        
        # Couche de sortie : 512 neurones en entrée, 'num_classes' en sortie (3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Passage dans la Couche 1
        x = self.pool1(self.relu1(self.conv1(x)))
        # Passage dans la Couche 2
        x = self.pool2(self.relu2(self.conv2(x)))
        # Passage dans la Couche 3
        x = self.pool3(self.relu3(self.conv3(x)))
        
        # Aplatir
        x = self.flatten(x)
        
        # Couches de décision
        x = self.relu4(self.fc1(x))
        x = self.fc2(x) # Pas de ReLU/Softmax ici, CrossEntropyLoss s'en charge
        return x

# --- 6. Initialisation de l'Entraînement ---
NUM_CLASSES = len(class_names)
model = SimpleCNN(num_classes=NUM_CLASSES).to(device)

# Fonction de perte (Loss Function) et Optimiseur
criterion = nn.CrossEntropyLoss() # Parfait pour la classification
optimizer = optim.Adam(model.parameters(), lr=0.001) 

NUM_EPOCHS = 15 # 15 époques, c'est bien pour un petit dataset

print("... Début de l'entraînement ...")

# --- 7. La Boucle d'Entraînement ---
for epoch in range(NUM_EPOCHS):
    
    # --- Phase d'entraînement ---
    model.train() # mode "entraînement"
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Mettre les données sur le bon "device" (CPU ou GPU)
        images, labels = images.to(device), labels.to(device)
        
        # 1. Remettre les gradients à zéro
        optimizer.zero_grad()
        
        # 2. Forward pass : Prédire les classes
        outputs = model(images)
        
        # 3. Calculer l'erreur (Loss)
        loss = criterion(outputs, labels)
        
        # 4. Backward pass : Rétro-propager l'erreur
        loss.backward()
        
        # 5. Mettre à jour les poids du réseau
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    
    # --- Phase de validation ---
    model.eval() # mode "évaluation"
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_val_loss = val_loss / len(val_dataset)
    accuracy = 100 * correct / total
    
    print(f"Époque [{epoch+1}/{NUM_EPOCHS}] - "
          f"Loss (Entraînement): {epoch_loss:.4f} - "
          f"Loss (Validation): {epoch_val_loss:.4f} - "
          f"Précision (Validation): {accuracy:.2f}%")

print("... Entraînement terminé ...")

# --- 8. Sauvegarder le Modèle ---
MODEL_SAVE_PATH = "hand_gesture_model.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"Modèle sauvegardé avec succès dans : {MODEL_SAVE_PATH}")
