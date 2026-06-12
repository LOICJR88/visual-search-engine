# Rapport - Moteur de Recherche Visuel

## 1. Architecture du système

Le système est composé de deux modules indépendants :

- **Module Encodeur (`encoder.py`)** : charge une image, applique le pré-traitement
  (resize 224x224, normalisation ImageNet), passe l'image dans un backbone
  (ResNet-50 sans couche fc, ou CLIP ViT-B/32) et retourne un vecteur d'embedding
  (2048 dims pour ResNet, 512 dims pour CLIP).

- **Module Requête (`query.py`)** : encode l'image requête avec le même modèle,
  interroge un index FAISS pré-construit, et retourne les K voisins les plus
  proches (chemin image, label, distance L2).

## 2. Données

Dataset : Clothing Dataset (alexeygrigorev/clothing-dataset), 5398 images
réparties sur 20 classes :

label
T-Shirt       1011
Longsleeve     699
Pants          692
Shoes          431
Shirt          378
Dress          357
Outwear        312
Shorts         308
Not sure       228
Hat            171
Skirt          155
Polo           120
Undershirt     118
Blazer         109
Hoodie         100
Body            69
Other           67
Top             43
Blouse          23
Skip             7

## 3. Index FAISS

- `index/resnet50_flat.index` : IndexFlatL2, recherche exacte, dim=2048
- `index/clip_flat.index` : IndexFlatL2, recherche exacte, dim=512
- `index/resnet50_ivf.index` : IndexIVFFlat (nlist=50, nprobe=5), recherche approximée

## 4. Résultats - Comparaison des encodeurs

| Encodeur       | Dimension | Precision@5 |
|----------------|-----------|-------------|
| ResNet-50      | 2048      | 0.624      |
| CLIP ViT-B/32  | 512       | 0.656      |

## 5. Visualisation des embeddings (t-SNE)

Voir `tsne_resnet.png` et `tsne_clip.png` : projection 2D des embeddings,
colorée par classe de vêtement. On observe que les clusters par classe sont
plus nettement séparés avec CLIP.

## 6. Analyse qualitative

- ResNet-50 capture davantage les motifs de texture/couleur bas niveau.
- CLIP (entraîné sur texte-image) capture davantage la sémantique globale
  de l'objet (type de vêtement), ce qui explique généralement une meilleure
  Precision@K sur ce type de tâche multi-classes.

## 7. Limites et améliorations possibles

- Index IVFFlat à affiner (augmenter nlist pour un plus grand dataset).
- Fine-tuning du modèle sur le dataset cible (apprentissage métrique / triplet loss).
- Ajout d'une recherche multimodale texte->image grâce à CLIP (recherche par description textuelle).
