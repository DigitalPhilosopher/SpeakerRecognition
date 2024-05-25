Cuda version 12.1 or higher needs to be installed

# Create virtual python environment 
python -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Add extraction utils
ln -s /path/to/extraction_utils source/extraction_utils

# Train Model

## Usage
usage: TrainModel.py [-h] [--downsample_train DOWNSAMPLE_TRAIN] [--downsample_valid DOWNSAMPLE_VALID] [--downsample_test DOWNSAMPLE_TEST]
                     --dataset DATASET [--mfccs MFCCS] [--sample_rate SAMPLE_RATE] --frontend FRONTEND [--frozen FROZEN]
                     [--embedding_size EMBEDDING_SIZE] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--validation_rate VALIDATION_RATE]
                     [--margin MARGIN] [--norm NORM] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--amsgrad AMSGRAD]
                     [--restart_epoch RESTART_EPOCH]

Training ECAPA-TDNN Model for Deepfake Speaker Verification

options:
  -h, --help            show this help message and exit
  --downsample_train DOWNSAMPLE_TRAIN
                        Downsample training data by a factor (default: 0 - no downsampling)
  --downsample_valid DOWNSAMPLE_VALID
                        Downsample validation data by a factor (default: 0 - no downsampling)
  --downsample_test DOWNSAMPLE_TEST
                        Downsample test data by a factor (default: 0 - no downsampling)
  --dataset DATASET     Which dataset to use (genuine | deepfake)
  --mfccs MFCCS         Number of MFCC features to extract (default: 13)
  --sample_rate SAMPLE_RATE
                        Sample rate for the audio data (default: 16000)
  --frontend FRONTEND   Which frontend model to use for feature extraction (mfcc, wavlm_base, wavlm_large)
  --frozen FROZEN       Whether the frontend model is jointly trained or frozen during training (1=frozen, 0=joint)
  --embedding_size EMBEDDING_SIZE
                        Size of the embedding vector (default: 192)
  --batch_size BATCH_SIZE
                        Batch size for training (default: 8)
  --epochs EPOCHS       Number of training epochs (default: 25)
  --validation_rate VALIDATION_RATE
                        Validation rate, i.e., validate every N epochs (default: 5)
  --margin MARGIN       Margin for loss function (default: 1)
  --norm NORM           Normalization type (default: 2)
  --learning_rate LEARNING_RATE
                        Learning rate for the optimizer (default: 0.001)
  --weight_decay WEIGHT_DECAY
                        Weight decay to use for optimizing (default: 0.00001)
  --amsgrad AMSGRAD     Whether to use the AMSGrad variant of Adam optimizer (default: False)
  --restart_epoch RESTART_EPOCH
                        Epoch at which to restart training (default: 5)


## Examples
python source/TrainModel.py --frontend mfcc --dataset genuine --batch_size 16 --epochs 20 --validation_rate 5 --margin 1 --restart_epoch 50 --mfccs 80 --downsample_valid 25 --downsample_test 50
python source/TrainModel.py --frontend mfcc --dataset deepfake --batch_size 16 --epochs 20 --validation_rate 5 --margin 1 --restart_epoch 50 --mfccs 80 --downsample_valid 25 --downsample_test 50

python source/TrainModel.py --frontend wavlm_base --dataset genuine --batch_size 8 --epochs 20 --validation_rate 5 --margin 1 --restart_epoch 50 --downsample_valid 25 --downsample_test 50
python source/TrainModel.py --frontend wavlm_base --dataset deepfake --batch_size 8 --epochs 20 --validation_rate 5 --margin 1 --restart_epoch 50 --downsample_valid 25 --downsample_test 50

python source/TrainModel.py --frontend wavlm_base --frozen 0 --dataset genuine --batch_size 8 --epochs 20 --validation_rate 5 --margin 1 --restart_epoch 50 --downsample_valid 25 --downsample_test 50
python source/TrainModel.py --frontend wavlm_base --frozen 0 --dataset deepfake --batch_size 8 --epochs 20 --validation_rate 5 --margin 1 --restart_epoch 50 --downsample_valid 25 --downsample_test 50

# Show results:
mlflow ui 

## Open in browser:
http://127.0.0.1:5000/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

# Inference

## Usage
usage: Inference.py [-h] --reference_audio REFERENCE_AUDIO --audio_in_question AUDIO_IN_QUESTION [--threshold THRESHOLD] --dataset DATASET
                    [--mfccs MFCCS] [--sample_rate SAMPLE_RATE] --frontend FRONTEND [--frozen FROZEN] [--embedding_size EMBEDDING_SIZE]

Inference of the ECAPA-TDNN Model for Deepfake Speaker Verification or Deepfake Detection

options:
  -h, --help            show this help message and exit
  --reference_audio REFERENCE_AUDIO
                        Genuine reference audio of speaker
  --audio_in_question AUDIO_IN_QUESTION
                        Audio in question to be speaker
  --threshold THRESHOLD
                        Threshold that can not be passed for it to beeing a genuine audio
  --dataset DATASET     Which dataset to use (genuine | deepfake)
  --mfccs MFCCS         Number of MFCC features to extract (default: 13)
  --sample_rate SAMPLE_RATE
                        Sample rate for the audio data (default: 16000)
  --frontend FRONTEND   Which frontend model to use for feature extraction (mfcc, wavlm_base, wavlm_large)
  --frozen FROZEN       Whether the frontend model is jointly trained or frozen during training (1=frozen, 0=joint)
  --embedding_size EMBEDDING_SIZE
                        Size of the embedding vector (default: 192)

## Examples
python source/Inference.py --frontend mfcc --dataset genuine --mfccs 80 --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav
python source/Inference.py --frontend mfcc --dataset deepfake --mfccs 80 --reference_audio ..data/reference.wav --audio_in_question ../data/question.wav

python source/Inference.py --frontend wavlm_base --dataset genuine --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav
python source/Inference.py --frontend wavlm_base --dataset deepfake --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav

python source/Inference.py --frontend wavlm_base --frozen 0 --dataset genuine  --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav
python source/Inference.py --frontend wavlm_base --frozen 0 --dataset deepfake --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav