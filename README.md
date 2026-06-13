# Moteur de Recherche Visuel (Image Query System)

Prototype de moteur de recherche par similarité d'images, basé sur des
embeddings extraits par ResNet-50 / CLIP ViT-B/32 et indexés avec FAISS.

## Structure du projet
visual_search_engine/

├── src/

│   ├── encoder.py      # Extraction des embeddings d'images

│   └── query.py         # Recherche des K images les plus similaires

├── index/

│   ├── resnet50_flat.index

│   ├── clip_flat.index

│   ├── resnet50_ivf.index

│   └── id_to_path.csv   # Mapping ID -> chemin image

├── embeddings/

│   └── comparison_summary.csv

│   ├──  visual_search_demo.ipynb

├── report/

│   ├── REPORT.md

│   ├── tsne_resnet.png

│   └── tsne_clip.png

├── images_sample/        # Images d'exemple pour tester

└── requirements.txt
## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Encodage des images

```bash
python src/encoder.py --image_dir clothing-dataset/images \
                       --csv clothing-dataset/images.csv \
                       --model resnet \
                       --output_dir embeddings
```

### 2. Recherche d'images similaires

```bash
python src/query.py --image images_sample/example.jpg \
                     --index index/resnet50_flat.index \
                     --mapping index/id_to_path.csv \
                     --model resnet \
                     --k 5
```

## Dataset

[Clothing Dataset](https://github.com/alexeygrigorev/clothing-dataset)
(alexeygrigorev/clothing-dataset)

## Résultats

Voir `report/REPORT.md` pour l'architecture détaillée, les métriques
Precision@K et la comparaison ResNet-50 vs CLIP.
