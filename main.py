import argparse
import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from config import TRAIN_DATASET, BATCH_SICE
from neural_net import get_model, train_model

def train():
    train_audio_dataset = AudioDataset(TRAIN_DATASET)
    train_loader = DataLoader(train_audio_dataset, batch_size=BATCH_SICE, shuffle=True)
    model = get_model()
    train_model(model, train_loader, len(set(train_audio_dataset.dataset['speaker'])), 192)
    torch.save(model, 'target/model.pth')

def main():
    parser = argparse.ArgumentParser(description="Model Operations")
    parser.add_argument('mode', choices=['train', 'evaluate', 'inference'], help='Mode of operation')
    
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'evaluate':
        pass
    elif args.mode == 'inference':
        pass

if __name__ == "__main__":
    main()
