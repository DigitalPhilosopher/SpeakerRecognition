from config import NUM_EPOCHS, MODEL, MODEL_PACKAGE, MODEL_PARAMS, LOSS, LOSS_PACKAGE, DEVICE
import importlib
import torch
import logging
import torch.nn as nn
from torch.utils.data import DataLoader

if DEVICE == 'CPU':
    device = torch.device("cpu")
    logging.info("Configured to use CPU.")
elif DEVICE.startswith('GPU'):
    if torch.cuda.is_available():
        if ':' in DEVICE:
            # Specific GPU requested
            gpu_index = int(DEVICE.split(':')[1])
            if gpu_index < torch.cuda.device_count():
                torch.cuda.set_device(gpu_index)
                device = torch.device(f"cuda:{gpu_index}")
                logging.info(f"Configured to use specific GPU: cuda:{gpu_index}.")
            else:
                logging.warning(f"Requested GPU index {gpu_index} is out of range. Defaulting to cuda:0.")
                device = torch.device("cuda:0")
        else:
            # General GPU request
            device = torch.device("cuda")
            logging.info("CUDA is available. Configured to use GPU.")
    else:
        logging.warning("CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
else:  # AUTO mode
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("CUDA is available. Using GPU (Auto mode).")
    else:
        device = torch.device("cpu")
        logging.info("CUDA is not available. Using CPU (Auto mode).")


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
    logging.debug(f"Attempting to import and initialize model {MODEL} from {MODEL_PACKAGE}.")
    try:
        model_module = importlib.import_module(f'.{MODEL}', package=MODEL_PACKAGE)
        Model = getattr(model_module, MODEL)
        model = Model(**MODEL_PARAMS)
        model.to(device)
        logging.info(f"Model {MODEL} initialized and moved to {device}.")
        return model
    except Exception as e:
        logging.error(f"Error initializing model {MODEL}: {e}", exc_info=True)
        raise

def train_model(model: nn.Module, data_loader: DataLoader, num_speakers: int, in_feats: int):
    """
    Trains a neural network model using the provided data loader for a predefined number of epochs.

    Parameters:
        data_loader (DataLoader): The DataLoader providing batches of input data and labels for training.
        model (nn.Module): The neural network model to be trained.
    """
    try:
        loss_module = importlib.import_module(f'.{LOSS}', package=LOSS_PACKAGE)
        LossClass = getattr(loss_module, LOSS)
        loss_function = LossClass(in_feats=in_feats, out_feats=num_speakers, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        logging.info(f"Training started for {NUM_EPOCHS} epochs.")

        for epoch in range(NUM_EPOCHS):
            logging.debug(f"Epoch {epoch+1}/{NUM_EPOCHS} started.")
            for features, labels in data_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                loss = loss_function(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} completed.")
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
        raise
