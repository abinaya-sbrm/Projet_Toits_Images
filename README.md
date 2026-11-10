# Détection de toits sur images satellitaires : effet du prétraitement

Projet de la matière « Analyse et restitution » (Sujet 2, proposé par Ikram Chourib).
Soutenance le 15/03/2027.

> README de cadrage (Séance 1, 10/11/2026). Les champs marqués **À COMPLÉTER** doivent être remplis par le binôme.

## 1. Problématique

Les images aériennes et satellitaires varient selon le capteur, l'heure, la saison et l'atmosphère. Un modèle entraîné sur une zone perd souvent en performance sur une autre.

**Question :** à modèle fixé (YOLO), quels prétraitements rendent les images homogènes et améliorent la détection de toits, surtout quand on change de zone ou de capteur ?

Notre contribution n'est pas un nouveau modèle. C'est la **preuve chiffrée** de ce que chaque prétraitement apporte, sur la zone d'entraînement et hors domaine.

## 2. Niveau choisi

**Segmentation d'instances.** Les annotations fournies sont des polygones
(format YOLO-seg), ce qui permet d'obtenir le contour de chaque toit, de
calculer les surfaces et de produire la carte demandée. Modèle d'ablation :
YOLO-seg (Ultralytics).



Remarque : les livrables 4 (erreur de surface) et 5 (carte des toits) demandent des masques. La segmentation (YOLO-seg) est donc la cible naturelle.

## 3. Données

| Jeu | Contenu | Accès |
|---|---|---|
| Dossier `dataset` fourni par l'enseignante | Orthophotos RGB à 20 cm/pixel, 1024 × 1024 px | Par l'enseignante |

- Nombre d'images : 364 tuiles RGB de 1024×1024 px (JPEG), réparties en train (291), val (36) et test (37)
- Zones/villes présentes : **À COMPLÉTER**
- Format des annotations (masques sémantiques ? polygones ?) : **À COMPLÉTER**
- Licence : **À COMPLÉTER**

Règles appliquées :
- découpage **par zone géographique**, jamais par tuile aléatoire (fuite spatiale) ;
- masques sémantiques → instances (composantes connexes) → polygones YOLO et COCO ;
- statistiques de normalisation calculées **sur l'entraînement uniquement** ;
- aucune image dans le dépôt (`data/` est vide, rempli par `scripts/download.py`).


- Nombre d'images : 364 (train 291, val 36, test 37), chacune avec son fichier d'annotation.
- Classe unique : `0: roof`.
- Format des annotations : polygones YOLO-seg (coordonnées normalisées).
- Nommage des fichiers : `nombre1_nombre2.jpg`, vraisemblablement des
  coordonnées de grille (colonne_ligne). Signification à confirmer auprès de l'enseignante.
- Emprise : à 20 cm/pixel, une image couvre environ 205 m × 205 m, soit environ 15 km² au total.
- Licence et provenance : à confirmer auprès de l'enseignante.
- Points de vigilance : nombreux toits coupés en bord d'image ; possibles
  toits mitoyens fusionnés en un seul polygone (à vérifier visuellement).

## 4. Zones source et cible

| Rôle | Zone | Nb d'images |
|---|---|---|
| Source (entraînement) | **À COMPLÉTER** | |
| Validation | **À COMPLÉTER** | |
| Cible (test hors domaine) | **À COMPLÉTER** | |
Le jeu ne contient pas de nom de ville. Les zones seront définies à partir des
coordonnées de grille des fichiers ; il reste à vérifier que le découpage
train/val/test fourni sépare bien des zones géographiques distinctes.
## 5. Rôles dans le binôme

| | Membre A « Données et prétraitement » | Membre B « Modèles et évaluation » |
|---|---|---|
| Nom | **Dufour Maxence** | **Abinaya Subraniam** |
| Données | Téléchargement, découpage par zone, conversion des annotations | Contrôle visuel, statistiques sur la taille des bâtiments |
| Prétraitement | Pipeline YAML, figures avant/après, tuilage et reconstitution | Augmentation de données |
| Modèles | SAM 3 | YOLO (ablation), Mask R-CNN, DETR |
| Évaluation | Erreur de surface, carte finale | mAP, IoU, écart source/cible, graines multiples |
| Restitution | Démonstration | Tableaux et figures |

En commun : ablation, interprétation, rapport, présentation. Une branche par fonctionnalité, chaque pull request est relue par l'autre membre, et `CONTRIBUTIONS.md` est mis à jour à chaque séance.

## 6. Installation

```bash
git clone <URL_DU_DEPOT>
cd toits-satellite-pretraitement
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 7. Premiers tests (Séance 1)

```bash
python scripts/test_tiling.py                                  # image synthétique
python scripts/test_tiling.py --image <chemin_image> --tile 512 --overlap 128
python -m pytest tests/
```

Une image 1024 × 1024 découpée en tuiles de 512 px avec 128 px de chevauchement donne 9 tuiles (coins à 0, 384 et 512 px). La reconstitution est vérifiée à l'identique.

## 8. Structure

```
configs/       une configuration YAML par expérience d'ablation
data/          vide, rempli par scripts/download.py
scripts/       téléchargement, découpage, conversion, tests
src/preprocessing/  tuilage, couleurs, contraste, histogrammes
src/models/    YOLO, Mask R-CNN, DETR, SAM 3
src/evaluation/  mAP, IoU, surface
notebooks/
results/       experiences.csv, tableaux, figures, carte
app/           démonstration Streamlit/Gradio
docs/          rapport, présentation
```

## 9. Planning

| Étape | Date | Attendu |
|---|---|---|
| Séance 1 | 10/11/2026 | Ce README de cadrage |
| Travail autonome | nov.–déc. | Conversion des annotations, pipeline, YOLO de référence |
| Séance 2 | 05/01/2027 | Pipeline complet, premiers résultats d'ablation |
| Travail autonome | janv.–févr. | Ablation, domaine, résolution, autres modèles, carte |
| Séance 3 | 01/03/2027 | Revue, test de la démonstration |
| Dépôt final | 12/03/2027 23h59 (à confirmer) | Tag v1.0 |
| Soutenance | 15/03/2027 | Présentation + démonstration |
