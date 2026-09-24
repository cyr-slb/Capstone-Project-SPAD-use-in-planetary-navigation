import cv2
import numpy as np

chemin_entree = r'C:\Users\cyria\Downloads\test1_60fps.mov'
chemin_sortie = r'C:\Users\cyria\Downloads\01_corrélationphase1.mp4'

# Permet de traiter une autre vidéo sans modifier le script :
#   python 01_corrélationphase1.py <video_entree> <video_sortie>
import sys
if len(sys.argv) >= 3:
    chemin_entree, chemin_sortie = sys.argv[-2], sys.argv[-1]


cap = cv2.VideoCapture(chemin_entree)
if not cap.isOpened():
    raise IOError(f"Impossible d'ouvrir : {chemin_entree}")

# Récupération des propriétés pour l'export
largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialisation de l'encodeur vidéo (MP4 avec codec H264/mp4v)
def source_en_niveaux_de_gris(chemin):
    """Vrai si la vidéo n'a pas de couleur (ses 3 canaux sont identiques).

    Réencoder une telle vidéo en couleur laisse un biais constant sur la
    chrominance : mesuré ici, +2.5 niveaux sur le vert. C'est invisible sur
    une image sombre, mais un étirement de contraste le multiplie et la
    vidéo vire au vert."""
    sonde = cv2.VideoCapture(chemin)
    ret, image = sonde.read()
    sonde.release()
    if not ret or image.ndim != 3:
        return True
    return bool(np.array_equal(image[:, :, 0], image[:, :, 1])
                and np.array_equal(image[:, :, 1], image[:, :, 2]))


EN_GRIS = source_en_niveaux_de_gris(chemin_entree)
if EN_GRIS:
    print("Source en niveaux de gris : la sortie le sera aussi.")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(chemin_sortie, fourcc, fps,
                      (largeur, hauteur), isColor=not EN_GRIS)

ret, frame_ref = cap.read()
if not ret:
    raise EOFError("Vidéo vide.")

# La première image sert de point zéro absolu, on l'écrit sans modification
out.write(frame_ref)

gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
# createHanningWindow attend (largeur, hauteur), pas l'ordre de shape
hanning_window = cv2.createHanningWindow((largeur, hauteur), cv2.CV_32F)

# Accumulateurs de trajectoire globale
dx_cumul = 0.0
dy_cumul = 0.0

nb_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
n = 0
print(f"Vidéo {largeur}x{hauteur} @ {fps:.2f} fps, {nb_total} images. Traitement...")


while True:
    ret, frame_curr = cap.read()
    if not ret:
        break

    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Calcul du décalage relatif entre t-1 et t
    shift, response = cv2.phaseCorrelate(gray_ref, gray_curr, window=hanning_window)
    dx_rel, dy_rel = shift
    
    # Accumulation pour obtenir la position par rapport à la première image (t=0)
    dx_cumul += dx_rel
    dy_cumul += dy_rel
    
    # Matrice de transformation affine 2x3 pour appliquer la translation inverse
    # On soustrait le décalage cumulé pour replacer l'image à l'origine
    M = np.float32([
        [1, 0, -dx_cumul],
        [0, 1, -dy_cumul]
    ])
    
    # Application de la matrice de translation
    frame_stabilisee = cv2.warpAffine(frame_curr, M, (largeur, hauteur))
    
    # Écriture dans le fichier de sortie
    if EN_GRIS:
        frame_stabilisee = cv2.cvtColor(frame_stabilisee, cv2.COLOR_BGR2GRAY)
    out.write(frame_stabilisee)
    
    # Mise à jour de l'image de référence pour le prochain calcul
    gray_ref = gray_curr

    n += 1
    if n % 30 == 0:
        print(f"  {n}/{nb_total} images - décalage cumulé ({dx_cumul:.1f}, {dy_cumul:.1f})", flush=True)

cap.release()
out.release()
print(f"Traitement terminé. Vidéo exportée : {chemin_sortie}")