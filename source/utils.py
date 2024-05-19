import torch
from extraction_utils.get_label_files import get_label_files
from tqdm.notebook import tqdm
import time
import mlflow
import mlflow.pytorch
import os
import warnings
import logging

def initialize_environment(model_name):
    global MODEL_NAME
    MODEL_NAME = model_name
    warnings.filterwarnings("ignore")
    logging.basicConfig(filename=MODEL_NAME + '.log', 
                        level=logging.INFO, 
                        format='%(asctime)s - %(message)s')
    global logger
    logger = logging.getLogger()
    return logger


def get_device():
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info("CUDA is available! Training on GPU...")
        device = torch.device("cuda")
    else:
        logger.info("CUDA is not available. Training on CPU...")
        device = torch.device("cpu")
    
    return device


def train_model_random_loss(epochs, dataloader, model, loss_function, optimizer, device):
    best_loss = float('inf')
    best_model_state = None

    with mlflow.start_run(run_name=MODEL_NAME):
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", dataloader.batch_size)
        mlflow.log_param("model", model.__class__.__name__)
        mlflow.log_param("loss_function", loss_function.__class__.__name__)
        mlflow.log_param("optimizer", optimizer.__class__.__name__)
        model.train()
        total_start_time = time.time()  # Start timing the whole training process
        
        for epoch in range(epochs):
            epoch_start_time = time.time()  # Start timing the epoch
            running_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for anchors, positives, negatives in progress_bar:
                batch_start_time = time.time()  # Start timing the batch
                
                anchors = anchors.to(device)
                positives = positives.to(device)
                negatives = negatives.to(device)

                optimizer.zero_grad()
                
                # Time the forward passes
                forward_start_time = time.time()
                anchor_outputs = model(anchors)
                positive_outputs = model(positives)
                negative_outputs = model(negatives)
                forward_end_time = time.time()
                
                loss = loss_function(anchor_outputs, positive_outputs, negative_outputs)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                
                batch_end_time = time.time()
                logger.info(f"Batch processed in {batch_end_time - batch_start_time:.4f} seconds.")
                logger.info(f"Forward pass took {forward_end_time - forward_start_time:.4f} seconds.")
           
                # Log GPU memory usage specific to this process
                if device.type == 'cuda':
                    memory_stats = torch.cuda.memory_stats(device)
                    gpu_memory_allocated = memory_stats["allocated_bytes.all.current"]
                    gpu_memory_reserved = memory_stats["reserved_bytes.all.current"]
                    mlflow.log_metric("gpu_memory_allocated", gpu_memory_allocated)
                    mlflow.log_metric("gpu_memory_reserved", gpu_memory_reserved)

            avg_loss = running_loss / len(dataloader)
            epoch_end_time = time.time()
            mlflow.log_metric("avg_loss", avg_loss, step=epoch)
            mlflow.log_metric("epoch_time", epoch_end_time - epoch_start_time, step=epoch)
            logger.info(f"Epoch {epoch+1} completed in {epoch_end_time - epoch_start_time:.4f} seconds. Average Loss: {avg_loss:.4f}")

            # Save the best model
            if avg_loss <= best_loss:
                best_loss = avg_loss
                best_model_state = model.state_dict()
                best_model_name = f"{MODEL_NAME}_best_model"
                mlflow.pytorch.log_model(model, best_model_name)

        total_end_time = time.time()
        mlflow.log_metric("total_training_time", total_end_time - total_start_time)
        logger.info(f"Training completed in {total_end_time - total_start_time:.4f} seconds.")
        print()

        # Save the latest model
        mlflow.pytorch.log_model(model, "latest_model")
        # Save the best model state as a model artifact
        if best_model_state is not None:
            best_model_state_filename = f"{MODEL_NAME}_best_model_state.pth"
            torch.save(best_model_state, best_model_state_filename)
            mlflow.log_artifact(best_model_state_filename)


def load_genuine_dataset():
    labels_text_path_list_train, labels_text_path_list_dev, labels_text_path_list_test, all_datasets_used = get_label_files(
        use_bsi_tts = False,
        use_bsi_vocoder = False,
        use_bsi_vc = False,
        use_bsi_genuine = True,
        use_bsi_ttsvctk = False,
        use_bsi_ttslj = False,
        use_bsi_ttsother = False,
        use_bsi_vocoderlj = False,
        use_wavefake = False,
        use_LibriSeVoc = False,
        use_lj = False,
        use_asv2019 = False,
    )
    return labels_text_path_list_train, labels_text_path_list_dev, labels_text_path_list_test


def load_deepfake_dataset():
    labels_text_path_list_train, labels_text_path_list_dev, labels_text_path_list_test, all_datasets_used = get_label_files(
        use_bsi_tts = True,
        use_bsi_vocoder = False,
        use_bsi_vc = False,
        use_bsi_genuine = True,
        use_bsi_ttsvctk = False,
        use_bsi_ttslj = False,
        use_bsi_ttsother = False,
        use_bsi_vocoderlj = False,
        use_wavefake = False,
        use_LibriSeVoc = False,
        use_lj = False,
        use_asv2019 = False,
    )
    return labels_text_path_list_train, labels_text_path_list_dev, labels_text_path_list_test


def clear_gpu():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
