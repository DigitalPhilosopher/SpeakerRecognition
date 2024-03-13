from config import N_MFCC, NUM_EPOCHS, MODEL, MODEL_PACKAGE, MODEL_PARAMS, LOSS, LOSS_PACKAGE
import importlib
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
    model_module = importlib.import_module(f'.{MODEL}', package=MODEL_PACKAGE)
    Model = getattr(model_module, MODEL)
    model = Model(**MODEL_PARAMS)
    model.to(device)
    return model

def train_model(model: nn.Module, data_loader: DataLoader, num_speakers: int, in_feats: int):
    """
    Trains a neural network model using the provided data loader for a predefined number of epochs.

    Parameters:
        data_loader (DataLoader): The DataLoader providing batches of input data and labels for training.
        model (nn.Module): The neural network model to be trained.
    """
    loss_module = importlib.import_module(f'.{LOSS}', package=LOSS_PACKAGE)
    LossClass = getattr(loss_module, LOSS)
    loss_function = LossClass(in_feats=in_feats, out_feats=num_speakers, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        for features, labels in data_loader:
            features = features.to(device)  # Assuming you're using a device like 'cuda' or 'cpu'
            labels = labels.to(device)
            # Forward pass
            outputs = model(features)
            # Compute loss
            loss = loss_function(outputs, labels)  # This is where both outputs and labels are used
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
