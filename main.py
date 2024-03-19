import argparse
import torch
import logging
from torch.utils.data import DataLoader
from dataset import AudioDataset
from config import TRAIN_DATASET, BATCH_SICE, MODEL, LOGGING_LEVEL
from neural_net import get_model, train_model

# Configure logging
logging.basicConfig(level=getattr(logging, LOGGING_LEVEL), format='%(asctime)s - %(levelname)s - %(message)s')

def train():
    logging.info("Training process started.")
    
    train_audio_dataset = AudioDataset(TRAIN_DATASET)
    train_loader = DataLoader(train_audio_dataset, batch_size=BATCH_SICE, shuffle=True)
    
    logging.info("Data loaded from " + TRAIN_DATASET + " and DataLoader initialized.")
    
    model = get_model()
    logging.info(f"Model {MODEL} initialized.")
    
    speaker_count = len(set(train_audio_dataset.dataset['speaker']))
    train_model(model, train_loader, speaker_count, 192)
    logging.info("Training completed.")
    
    torch.save(model, f'target/{MODEL}.pth')
    logging.info(f"Model saved to 'target/{MODEL}.pth'.")

def main():
    parser = argparse.ArgumentParser(description="Model Operations")
    parser.add_argument('mode', choices=['train', 'evaluate', 'inference'], help='Mode of operation')
    
    args = parser.parse_args()

    logging.info(f"Operation mode: {args.mode}")
    
    if args.mode == 'train':
        train()
    elif args.mode == 'evaluate':
        logging.info("Evaluation mode is not yet implemented.")
        # Implement evaluate functionality
    elif args.mode == 'inference':
        logging.info("Inference mode is not yet implemented.")
        # Implement inference functionality

if __name__ == "__main__":
    main()
