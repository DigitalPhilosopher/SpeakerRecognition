from config import N_MFCC, NUM_EPOCHS
from model import Model, Loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_model() -> nn.Module:
    """
    Creates and returns a model instance configured for either GPU or CPU based on CUDA availability.

    This function initializes a model with predefined parameters: input size (N_MFCC), hidden layer size (128),
    number of layers (2), and number of classes determined by the unique speakers in the training dataset.
    It then moves the model to GPU if CUDA is available, otherwise to CPU.

    The input size (N_MFCC) should be defined globally or within the scope accessible to this function.
    The `train_audio_dataset` should be a dataset or a dataloader object containing 'speaker' labels
    to dynamically determine the number of output classes for the model based on unique speakers.

    Returns:
        nn.Module: The initialized model, placed on the appropriate device (GPU or CPU).
    """
    model = Model(input_size=N_MFCC, hidden_size=128, num_layers=2, embedding_dim=64)
    model.to(device)
    return model

def train_model(model: nn.Module, data_loader: DataLoader, num_speakers: int, num_utterances: int):
    """
    Trains a neural network model using the provided data loader for a predefined number of epochs.

    Parameters:
        data_loader (DataLoader): The DataLoader providing batches of input data and labels for training.
        model (nn.Module): The neural network model to be trained.
    """
    loss = Loss(init_w=10.0, init_b=-5.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        for batch in data_loader:  # Your DataLoader needs to provide batches structured for GE2E
            embeddings = model(batch)  # Forward pass to get embeddings
            loss = loss(embeddings, num_speakers, num_utterances)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')