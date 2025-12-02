import cv2 as cv
import numpy as np

# Initialisation de la capture vidéo
cap = cv.VideoCapture(0)

# Variable pour stocker l'arrière-plan (image de référence)
background = None

# PARAMÈTRES COULEUR (issus de la Partie 1 du notebook)
color_ranges = {
    'rouge': [
        (np.array([0, 40, 40]), np.array([10, 255, 255])),
        (np.array([170, 40, 40]), np.array([180, 255, 255]))
    ],
    'vert': [
        (np.array([35, 50, 50]), np.array([85, 255, 255]))
    ],
    'bleu': [
        (np.array([85, 30, 30]), np.array([130, 255, 255]))
    ]
}

couleur_active = 'rouge'  # Couleur à détecter

# PARAMÈTRES DE DÉTECTION
SEUIL_MOUVEMENT = 25      # Seuil pour la soustraction de fond
KERNEL_SIZE = 5           # Taille du noyau morphologique

kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

print("=" * 60)
print("DÉTECTION D'OBJETS EN MOUVEMENT")
print("=" * 60)
print(f"Couleur active : {couleur_active.upper()}")
print("\nCommandes :")
print("  - ESC : Quitter")
print("  - 'r' : Détecter ROUGE")
print("  - 'v' : Détecter VERT")
print("  - 'b' : Détecter BLEU")


while True:
    # Lecture de l'image courante
    ret, frame = cap.read()
    if not ret:
        print("ERREUR : Impossible de lire la frame")
        break

    # Flou gaussien pour réduire le bruit (comme dans le notebook)
    frame_blurred = cv.GaussianBlur(frame, (5, 5), 0)
    
    # Conversion en niveaux de gris (pour la détection de mouvement)
    gray = cv.cvtColor(frame_blurred, cv.COLOR_BGR2GRAY)
    
    # --- ÉTAPE 1 : Initialiser le fond (au premier tour seulement) ---
    if background is None:
        background = gray.copy()
        print("✓ Fond de référence capturé")
        continue

    # --- ÉTAPE 2 : DÉTECTION DE MOUVEMENT (Soustraction de fond) ---
    # Calculer la différence avec le fond
    diff = cv.absdiff(background, gray)
    
    # Seuillage : Si différence > SEUIL, c'est du mouvement
    _, mask_mouvement = cv.threshold(diff, SEUIL_MOUVEMENT, 255, cv.THRESH_BINARY)
    
    # Nettoyage morphologique (comme dans le notebook)
    # Opening pour supprimer le bruit
    mask_mouvement = cv.morphologyEx(mask_mouvement, cv.MORPH_OPEN, kernel)
    # Closing pour remplir les trous
    mask_mouvement = cv.morphologyEx(mask_mouvement, cv.MORPH_CLOSE, kernel)

    # --- ÉTAPE 3 : DÉTECTION DE COULEUR (Segmentation HSV) ---
    # Conversion HSV (comme dans la Partie 1)
    hsv = cv.cvtColor(frame_blurred, cv.COLOR_BGR2HSV)
    
    # Création du masque couleur selon la couleur active
    mask_couleur = np.zeros(gray.shape, dtype=np.uint8)
    
    # Appliquer les plages HSV de la Partie 1
    for lower, upper in color_ranges[couleur_active]:
        mask_temp = cv.inRange(hsv, lower, upper)
        mask_couleur = cv.add(mask_couleur, mask_temp)
    
    # Nettoyage morphologique du masque couleur (comme dans le notebook)
    mask_couleur = cv.morphologyEx(mask_couleur, cv.MORPH_OPEN, kernel)
    mask_couleur = cv.morphologyEx(mask_couleur, cv.MORPH_CLOSE, kernel)

    # --- ÉTAPE 4 : FUSION (Mouvement ET Couleur) ---
    # On garde seulement les pixels qui sont BLANCS dans les DEUX masques
    mask_final = cv.bitwise_and(mask_mouvement, mask_couleur)

    # --- ÉTAPE 5 : AFFICHAGE DU RÉSULTAT ---
    # Appliquer le masque final sur l'image originale
    resultat = cv.bitwise_and(frame, frame, mask=mask_final)
    cv.imshow('A - Masque Mouvement', mask_mouvement)
    cv.imshow('B - Masque Couleur', mask_couleur)
    cv.imshow('C - Masque Final', mask_final)
    # Afficher le résultat
    cv.imshow('RESULTAT', resultat)

    # --- GESTION DES TOUCHES ---
    k = cv.waitKey(5) & 0xFF
    
    if k == 27:  # ESC pour quitter
        print("\n✓ Arrêt demandé")
        break
    elif k == ord('r'):
        couleur_active = 'rouge'
        print(f"✓ Détection : ROUGE")
    elif k == ord('v'):
        couleur_active = 'vert'
        print(f"✓ Détection : VERT")
    elif k == ord('b'):
        couleur_active = 'bleu'
        print(f"✓ Détection : BLEU")

# Libération des ressources
cap.release()
cv.destroyAllWindows()
print("✓ Programme terminé")