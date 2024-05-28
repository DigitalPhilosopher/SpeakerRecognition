# Audio Deepfake Detection with the aid of Authentic Reference Material

This repository was developed as part of the research for my Master's thesis titled "Audio Deepfake Detection with the Aid of Authentic Reference Material," conducted at the University of Hagen. The thesis was supervised by Prof. Jörg Keller ([University of Hagen](https://www.fernuni-hagen.de/)) and Dr. Dominique Dresen ([Federal Office for Information Security](https://www.bsi.bund.de/)), with additional support from Matthias Neu ([Federal Office for Information Security](https://www.bsi.bund.de/)).

The primary objective of this repository is to facilitate the development of a deepfake detection model using the ECAPA-TDNN [1] architecture. The audio features are generated using either an MFCC extractor or the SSL model WavLM [2]. A key innovation in this thesis is the introduction of the triplet loss function for training the model, as opposed to the current state-of-the-art which uses ECAPA-TDNN with WavLM-Large but employs the AAM-Softmax loss function [3].

To achieve better results in deepfake detection, the triplets are generated as anchor A (audio of speaker A), positive P (different audio of speaker A), and deepfake D (deepfake of speaker A). The loss function aims to minimize the Euclidean distance of the embeddings generated from the ECAPA-TDNN model in the same manner as done in FaceNet [4].

To create these triplets, a dataset is required that consists of authentic audios as well as deepfake audios of the same speaker. The Federal Office for Information Security provided a dataset including authentic audio files from LibriTTS [5] and their corresponding deepfakes. These deepfakes are produced using both Text-to-Speech (TTS) and Voice Conversion (VC) methods. An extensive list of deepfake methods used to generate the dataset is listed in [Deepfake Methods](#deepfake-methods)


## Setup

This repository leverages the deepfake dataset provided by the Federal Office for Information Security. The dataset is accompanied by extraction code located in the `extraction_utils` directory. To properly utilize this code, it is necessary to create a symbolic link in the `./source` folder.

This project requires CUDA-enabled graphics cards for execution. Ensure that CUDA version 12.1 or higher is installed on your system. You can verify your CUDA installation by running:
```bash
nvcc --version
```
If you do not have the required CUDA version, follow the installation instructions provided by NVIDIA to upgrade or install CUDA version 12.1 or higher.

### Steps to initialize the project

1. Download the project
```bash
git clone git@github.com:DigitalPhilosopher/SpeakerRecognition.git
```

2. Create virtual python environment
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install pip requirements
```bash
pip install -r requirements.txt
```

4. Add a symbolic link to extraction utils (This is the code given by the Federal Office for Information Security)
```bash
ln -s /path/to/extraction_utils source/extraction_utils
```

# Train Model

This script is designed to train a specific model using the provided deepfake dataset. Users can modify various parameters to adjust hyperparameters and training functionalities according to their requirements. To train the model, execute the script with the desired parameters. The script allows for flexibility in setting different hyperparameters, such as learning rate, batch size, and number of epochs, among others.

```
usage: TrainModel.py [-h] [--downsample_train DOWNSAMPLE_TRAIN] [--downsample_valid DOWNSAMPLE_VALID] [--downsample_test DOWNSAMPLE_TEST] --dataset DATASET
                     [--mfccs MFCCS] [--sample_rate SAMPLE_RATE] --frontend FRONTEND [--frozen FROZEN] [--embedding_size EMBEDDING_SIZE] [--device DEVICE]
                     [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--validation_rate VALIDATION_RATE] [--margin MARGIN] [--norm NORM] [--learning_rate LEARNING_RATE]
                     [--weight_decay WEIGHT_DECAY] [--amsgrad AMSGRAD]

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
  --device DEVICE       Which device to use (per default looks if cuda is available)
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
```

**Examples**

Below are some examples to help you get started with training the model using different configurations:


```bash
# Frontend: MFCC
python source/TrainModel.py --frontend mfcc --dataset genuine --batch_size 16 --epochs 20 --validation_rate 5 --margin 0.2 --mfccs 80 --downsample_valid 25 --downsample_test 50
python source/TrainModel.py --frontend mfcc --dataset deepfake --batch_size 16 --epochs 20 --validation_rate 5 --margin 0.2 --mfccs 80 --downsample_valid 25 --downsample_test 50

# Frontend: WavLM Base with frozen parameters
python source/TrainModel.py --frontend wavlm_base --dataset genuine --batch_size 8 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50
python source/TrainModel.py --frontend wavlm_base --dataset deepfake --batch_size 8 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50

# Frontend: WavLM Base with jointly trained parameters
python source/TrainModel.py --frontend wavlm_base --frozen 0 --dataset genuine --batch_size 8 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50
python source/TrainModel.py --frontend wavlm_base --frozen 0 --dataset deepfake --batch_size 8 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50

# Frontend: WavLM Large with frozen parameters
python source/TrainModel.py --frontend wavlm_large --dataset genuine --batch_size 8 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50
python source/TrainModel.py --frontend wavlm_large --dataset deepfake --batch_size 8 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50

# Frontend: WavLM Base with jointly trained parameters
python source/TrainModel.py --frontend wavlm_large --frozen 0 --dataset genuine --batch_size 4 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50
python source/TrainModel.py --frontend wavlm_large --frozen 0 --dataset deepfake --batch_size 4 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50
```

**Display training results**

To visualize and monitor training results, this project utilizes MLflow. MLflow provides a robust interface for tracking experiments, visualizing metrics, and managing model artifacts. To start the MLflow UI and navigate to the local interface, execute the following command in your terminal:
```bash
mlflow ui
```
After running the command, open your web browser and go to [MLflow Page on localhost](http://127.0.0.1:5000).


# Inference

To perform inference using the ECAPA-TDNN Model for Deepfake Speaker Verification or Deepfake Detection, utilize the script for inference. This script allows you to compare a reference audio file with an audio file in question to determine if the latter is genuine or a deepfake.

```bash
usage: Inference.py [-h] --reference_audio REFERENCE_AUDIO --audio_in_question AUDIO_IN_QUESTION [--threshold THRESHOLD] --dataset DATASET [--mfccs MFCCS]
                    [--sample_rate SAMPLE_RATE] --frontend FRONTEND [--frozen FROZEN] [--embedding_size EMBEDDING_SIZE] [--device DEVICE]

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
  --device DEVICE       Which device to use (per default looks if cuda is available)
```

**Examples**

Below are some examples to help you get started with basic inference on the trained models using different configurations:

```bash
# Frontend: MFCC
python source/Inference.py --frontend mfcc --dataset genuine --mfccs 80 --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav
python source/Inference.py --frontend mfcc --dataset deepfake --mfccs 80 --reference_audio ..data/reference.wav --audio_in_question ../data/question.wav

# Frontend: WavLM Base with frozen parameters
python source/Inference.py --frontend wavlm_base --dataset genuine --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav
python source/Inference.py --frontend wavlm_base --dataset deepfake --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav

# Frontend: WavLM Base with jointly trained parameters
python source/Inference.py --frontend wavlm_base --frozen 0 --dataset genuine  --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav
python source/Inference.py --frontend wavlm_base --frozen 0 --dataset deepfake --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav

# Frontend: WavLM Large with frozen parameters
python source/Inference.py --frontend wavlm_large --dataset genuine --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav
python source/Inference.py --frontend wavlm_large --dataset deepfake --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav

# Frontend: WavLM Large with jointly trained parameters
python source/Inference.py --frontend wavlm_large --frozen 0 --dataset genuine  --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav
python source/Inference.py --frontend wavlm_large --frozen 0 --dataset deepfake --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav
```


# Analytics

This script performs analytics on the trained models and saves the results to the `./data/analytics.csv` file. It calculates the Equal Error Rate (EER) and the minimum Detection Cost Function (minDCF), and generates a threshold at which the EER is minimized.

```bash
usage: Analytics.py [-h] [--train | --no-train] [--valid | --no-valid] [--test | --no-test] [--downsample_train DOWNSAMPLE_TRAIN]
                    [--downsample_valid DOWNSAMPLE_VALID] [--downsample_test DOWNSAMPLE_TEST] --dataset DATASET [--mfccs MFCCS] [--sample_rate SAMPLE_RATE]
                    --frontend FRONTEND [--frozen FROZEN] [--embedding_size EMBEDDING_SIZE] [--device DEVICE] [--batch_size BATCH_SIZE]
                    [--analyze_genuine | --no-analyze_genuine] [--analyze_deepfake | --no-analyze_deepfake]

Analytics of the ECAPA-TDNN Model for Deepfake Speaker Verification and Deepfake Detection

options:
  -h, --help            show this help message and exit
  --train, --no-train   Whether to generate analytics for the training set (default=False) (default: False)
  --valid, --no-valid   Whether to generate analytics for the valid set (default=False) (default: False)
  --test, --no-test     Whether to generate analytics for the test set (default=True) (default: True)
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
  --device DEVICE       Which device to use (per default looks if cuda is available)
  --batch_size BATCH_SIZE
                        Batch size for training (default: 8)
  --analyze_genuine, --no-analyze_genuine
                        Whether to generate analytics for the genuine dataset. (default: True)
  --analyze_deepfake, --no-analyze_deepfake
                        Whether to generate analytics for the deepfake dataset. (default: True)
```

**Examples**

Below are some examples to help you get started with starting the analytics on the trained models:

```bash
# Frontend: MFCC
python source/Analytics.py --frontend mfcc --dataset genuine --mfccs 80 --batch_size 16 --downsample_train 1000
python source/Analytics.py --frontend mfcc --dataset deepfake --mfccs 80 --batch_size 16 --downsample_train 1000

# Frontend: WavLM Base with frozen parameters
python source/Analytics.py --frontend wavlm_base --dataset genuine --batch_size 8 --downsample_train 1000
python source/Analytics.py --frontend wavlm_base --dataset deepfake --batch_size 8 --downsample_train 1000

# Frontend: WavLM Base with jointly trained parameters
python source/Analytics.py --frontend wavlm_base --frozen 0 --dataset genuine --batch_size 8 --downsample_train 1000
python source/Analytics.py --frontend wavlm_base --frozen 0 --dataset deepfake --batch_size 8 --downsample_train 1000

# Frontend: WavLM Large with frozen parameters
python source/Analytics.py --frontend wavlm_large --dataset genuine --batch_size 8 --downsample_train 1000
python source/Analytics.py --frontend wavlm_large --dataset deepfake --batch_size 8 --downsample_train 1000

# Frontend: WavLM Large with jointly trained parameters
python source/Analytics.py --frontend wavlm_large --frozen 0 --dataset genuine --batch_size 8 --downsample_train 1000
python source/Analytics.py --frontend wavlm_large --frozen 0 --dataset deepfake --batch_size 8 --downsample_train 1000
```


# Experiments

The experiments script can be used to run several different experiments at once. This script reads a text file, where each line represents an experiment. It will check how many GPU's are available and run experiments on all of them. To make sure every model is trained before analytics are run, it will first run training scripts, just after all these scripts are finished, it will run inference and finally analytics scripts. For testing, the a [Lightweight set](./experiments_light.txt) was added as example. To train and analyze all the models, the [Full set](experiments.txt) was added.

```bash
usage: Experiments.py [-h] [--experiments EXPERIMENTS]

Running several experiments on ECAPA-TDNN Model for Deepfake Speaker Verification

options:
  -h, --help            show this help message and exit
  --experiments EXPERIMENTS
                        File of experiments to run
```

**Example**
```bash
python source/Experiments.py --experiments experiments.txt
```
