
"""
encoder.py - Module d'encodage d'images pour le moteur de recherche visuel.
Usage:
    python encoder.py --image_dir clothing-dataset/images --csv clothing-dataset/images.csv --model resnet
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from tqdm import tqdm
import open_clip


class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['filepath']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, idx


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


@torch.no_grad()
def extract_embeddings(model, dataloader, device, model_type='resnet'):
    all_embeddings = []
    for images, _ in tqdm(dataloader):
        images = images.to(device)
        if model_type == 'clip':
            features = model.encode_image(images)
        else:
            features = model(images)
        all_embeddings.append(features.cpu().numpy())
    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'clip'])
    parser.add_argument('--output_dir', type=str, default='embeddings')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(args.csv)
    df['filepath'] = df['image'].apply(lambda x: os.path.join(args.image_dir, x + '.jpg'))
    df = df[df['filepath'].apply(os.path.exists)].reset_index(drop=True)

    if args.model == 'clip':
        model, preprocess = load_clip(device)
    else:
        model, preprocess = load_resnet(device)

    dataset = ImageDataset(df, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    embeddings = extract_embeddings(model, dataloader, device, model_type=args.model)

    np.save(os.path.join(args.output_dir, f'{args.model}_embeddings.npy'), embeddings)
    df.to_csv(os.path.join(args.output_dir, 'image_metadata.csv'), index=False)

    print(f"Embeddings sauvegardés : {embeddings.shape} -> {args.output_dir}/{args.model}_embeddings.npy")


if __name__ == '__main__':
    main()
