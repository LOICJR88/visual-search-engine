
"""
query.py - Module de requête pour le moteur de recherche visuel.
Usage:
    python query.py --image path/to/query.jpg --index index/resnet50_flat.index \
                     --mapping index/id_to_path.csv --model resnet --k 5
"""

import argparse
import numpy as np
import pandas as pd
import faiss
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import open_clip


def load_resnet(device):
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])
    return model, preprocess


def load_clip(device):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model.eval().to(device)
    return model, preprocess


def encode_image(image_path, model, preprocess, device, model_type='resnet'):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type == 'clip':
            features = model.encode_image(img_tensor)
        else:
            features = model(img_tensor)

    return features.cpu().numpy().astype('float32')


def search(query_vector, index, mapping_df, k=5):
    distances, indices = index.search(query_vector, k)
    results = mapping_df.iloc[indices[0]].copy()
    results['distance'] = distances[0]
    return results.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Chemin de l\'image requête')
    parser.add_argument('--index', type=str, required=True, help='Chemin de l\'index FAISS')
    parser.add_argument('--mapping', type=str, required=True, help='CSV id_to_path')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'clip'])
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'clip':
        model, preprocess = load_clip(device)
    else:
        model, preprocess = load_resnet(device)

    index = faiss.read_index(args.index)
    mapping_df = pd.read_csv(args.mapping)

    query_vec = encode_image(args.image, model, preprocess, device, model_type=args.model)
    results = search(query_vec, index, mapping_df, k=args.k)

    print(f"\nTop {args.k} résultats pour {args.image} :\n")
    for _, row in results.iterrows():
        print(f"  {row['filepath']:50s}  label={row['label']:15s}  distance={row['distance']:.4f}")


if __name__ == '__main__':
    main()
