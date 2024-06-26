# Audio Deepfake Detection with the aid of Authentic Reference Material

This repository was developed as part of the research for my Master's thesis titled "Audio Deepfake Detection with the Aid of Authentic Reference Material," conducted at the University of Hagen. The thesis was supervised by Prof. Jörg Keller ([University of Hagen](https://www.fernuni-hagen.de/)) and Dr. Dominique Dresen ([Federal Office for Information Security](https://www.bsi.bund.de/)), with additional support from Matthias Neu ([Federal Office for Information Security](https://www.bsi.bund.de/)).

The primary objective of this repository is to facilitate the development of a deepfake detection model using the ECAPA-TDNN [1] architecture (see Fig. 1). The audio features are generated using either an MFCC extractor or the SSL model WavLM [2]. A key innovation in this thesis is the introduction of the triplet loss function for training the model, as opposed to the current state-of-the-art which uses ECAPA-TDNN with WavLM-Large but employs the AAM-Softmax loss function [3]. To have standarized tested code, this repository uses the ECAPA-TDNN implementation of [SpeechBrain](https://github.com/speechbrain/speechbrain)[4]. The code can be found in their GitHub repository at [speechbrain/lobes/models/ECAPA_TDNN.py](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/ECAPA_TDNN.py). Additionally, [S3PRL](https://github.com/s3prl/s3prl)[5] is used as speech toolkit, to leverage WavLm as feature generator. To leverage additional upstream ssl models included in S3PRL, the abstract [s3prl_ECAPA_TDNN](source/models/s3prl_ECAPA_TDNN.py) class can be implemented. Here we implemented [WavLM Base](source/models/WavLM_Base_ECAPA_TDNN.py) as well as [WavLM Large](source/models/WavLM_Large_ECAPA_TDNN.py).

<p align="center">
    <a href="images/ECAPA-TDNN.ppm">
        <img src="images/ECAPA-TDNN.ppm" width="500"/>
    </a>
    <p align="center"><em>Fig. 1 ECAPA-TDNN architecture including the SE-Res2Block [1].</em></p>
</p>


To achieve better results in deepfake detection, the triplets are generated as anchor A (audio of speaker A), positive P (different audio of speaker A), and deepfake D (deepfake of speaker A). The loss function aims to minimize the Euclidean distance of the embeddings generated from the ECAPA-TDNN model in the same manner as done in FaceNet [6]. The loss is calculated using the following function:
$$\mathcal{L}(A, P, D) = \max(\|f(A) - f(P)\|_2^2 - \|f(A) - f(D)\|_2^2 + \alpha, 0)$$
where $f$ represents the embedding function generated by the ECAPA-TDNN model, and $\alpha$ is a margin that ensures the deepfake is farther from the anchor than the positive. The triplet loss is calculated using the [TripletMarginWithDistanceLoss](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html#torch.nn.TripletMarginWithDistanceLoss)[7] from PyTorch using the [compute_distance](source/utils/distance.py) function, which calculates the squared Euclidean distance given two L2-normalized vectors.

To create these triplets, a dataset is required that consists of authentic audios as well as deepfake audios of the same speaker. The Federal Office for Information Security provided a dataset including authentic audio files from LibriTTS [8] and their corresponding deepfakes. These deepfakes are produced using both Text-to-Speech (TTS) and Voice Conversion (VC) methods. An extensive list of deepfake methods used to generate the dataset is listed in [Deepfake Methods](#deepfake-methods)


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

4. Add a symbolic link or copy data to data dir
```bash
ln -s /path/to/BSI_DATASET data/BSI
ln -s /path/to/LibriSpeech data/LibriSpeech
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
  --dataset DATASET     Which dataset to use (LibriSpeech.genuine | VoxCeleb.genuine | BSI.genuine | BSI.deepfake)
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
python source/TrainModel.py --frontend mfcc --dataset BSI.genuine --batch_size 16 --epochs 20 --validation_rate 5 --margin 0.2 --mfccs 80 --downsample_valid 25 --downsample_test 50
python source/TrainModel.py --frontend mfcc --dataset BSI.deepfake --batch_size 16 --epochs 20 --validation_rate 5 --margin 0.2 --mfccs 80 --downsample_valid 25 --downsample_test 50

# Frontend: WavLM Base with frozen parameters
python source/TrainModel.py --frontend wavlm_base --dataset BSI.genuine --batch_size 8 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50
python source/TrainModel.py --frontend wavlm_base --dataset BSI.deepfake --batch_size 8 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50

# Frontend: WavLM Base with jointly trained parameters
python source/TrainModel.py --frontend wavlm_base --frozen 0 --dataset BSI.genuine --batch_size 8 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50
python source/TrainModel.py --frontend wavlm_base --frozen 0 --dataset BSI.deepfake --batch_size 8 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50

# Frontend: WavLM Large with frozen parameters
python source/TrainModel.py --frontend wavlm_large --dataset BSI.genuine --batch_size 8 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50
python source/TrainModel.py --frontend wavlm_large --dataset BSI.deepfake --batch_size 8 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50

# Frontend: WavLM Base with jointly trained parameters
python source/TrainModel.py --frontend wavlm_large --frozen 0 --dataset BSI.genuine --batch_size 4 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50
python source/TrainModel.py --frontend wavlm_large --frozen 0 --dataset BSI.deepfake --batch_size 4 --epochs 20 --validation_rate 5 --margin 0.2 --downsample_valid 25 --downsample_test 50
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
  --dataset DATASET     Which dataset to use (LibriSpeech.genuine | VoxCeleb.genuine | BSI.genuine | BSI.deepfake)
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
python source/Inference.py --frontend mfcc --dataset BSI.genuine --mfccs 80 --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav
python source/Inference.py --frontend mfcc --dataset BSI.deepfake --mfccs 80 --reference_audio ..data/reference.wav --audio_in_question ../data/question.wav

# Frontend: WavLM Base with frozen parameters
python source/Inference.py --frontend wavlm_base --dataset BSI.genuine --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav
python source/Inference.py --frontend wavlm_base --dataset BSI.deepfake --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav

# Frontend: WavLM Base with jointly trained parameters
python source/Inference.py --frontend wavlm_base --frozen 0 --dataset BSI.genuine  --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav
python source/Inference.py --frontend wavlm_base --frozen 0 --dataset BSI.deepfake --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav

# Frontend: WavLM Large with frozen parameters
python source/Inference.py --frontend wavlm_large --dataset BSI.genuine --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav
python source/Inference.py --frontend wavlm_large --dataset BSI.deepfake --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav

# Frontend: WavLM Large with jointly trained parameters
python source/Inference.py --frontend wavlm_large --frozen 0 --dataset BSI.genuine  --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav
python source/Inference.py --frontend wavlm_large --frozen 0 --dataset BSI.deepfake --reference_audio ../data/reference.wav --audio_in_question ../data/question.wav
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
  --dataset DATASET     Which dataset to use (LibriSpeech.genuine | VoxCeleb.genuine | BSI.genuine | BSI.deepfake)
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
python source/Analytics.py --frontend mfcc --dataset BSI.genuine --mfccs 80 --batch_size 16 --downsample_train 1000
python source/Analytics.py --frontend mfcc --dataset BSI.deepfake --mfccs 80 --batch_size 16 --downsample_train 1000

# Frontend: WavLM Base with frozen parameters
python source/Analytics.py --frontend wavlm_base --dataset BSI.genuine --batch_size 8 --downsample_train 1000
python source/Analytics.py --frontend wavlm_base --dataset BSI.deepfake --batch_size 8 --downsample_train 1000

# Frontend: WavLM Base with jointly trained parameters
python source/Analytics.py --frontend wavlm_base --frozen 0 --dataset BSI.genuine --batch_size 8 --downsample_train 1000
python source/Analytics.py --frontend wavlm_base --frozen 0 --dataset BSI.deepfake --batch_size 8 --downsample_train 1000

# Frontend: WavLM Large with frozen parameters
python source/Analytics.py --frontend wavlm_large --dataset BSI.genuine --batch_size 8 --downsample_train 1000
python source/Analytics.py --frontend wavlm_large --dataset BSI.deepfake --batch_size 8 --downsample_train 1000

# Frontend: WavLM Large with jointly trained parameters
python source/Analytics.py --frontend wavlm_large --frozen 0 --dataset BSI.genuine --batch_size 8 --downsample_train 1000
python source/Analytics.py --frontend wavlm_large --frozen 0 --dataset BSI.deepfake --batch_size 8 --downsample_train 1000
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

# Results

## First Models: BSI-Dataset

### Speaker Verification
| Front-End          | Triplet Mining   | Dataset    |   Speaker Verification EER |
|:-------------------|:-----------------|:-----------|---------------------------:|
| MFCC               | Deepfake         | training   |                   0.727663 |
| MFCC               | Deepfake         | validation |                   0.721183 |
| MFCC               | Deepfake         | test       |                   0.728241 |
| MFCC               | Genuine          | training   |                   0.701033 |
| MFCC               | Genuine          | validation |                   0.701497 |
| MFCC               | Genuine          | test       |                   0.707249 |
| WavLM-Base/Frozen  | Deepfake         | training   |                   0.669932 |
| WavLM-Base/Frozen  | Deepfake         | validation |                   0.667618 |
| WavLM-Base/Frozen  | Deepfake         | test       |                   0.671599 |
| WavLM-Base/Frozen  | Genuine          | training   |                   0.711452 |
| WavLM-Base/Frozen  | Genuine          | validation |                   0.710852 |
| WavLM-Base/Frozen  | Genuine          | test       |                   0.713652 |
| WavLM-Base/Joint   | Deepfake         | training   |                   0.676053 |
| WavLM-Base/Joint   | Deepfake         | validation |                   0.6809   |
| WavLM-Base/Joint   | Deepfake         | test       |                   0.679986 |
| WavLM-Base/Joint   | Genuine          | training   |               **0.563248** |
| WavLM-Base/Joint   | Genuine          | validation |               **0.565732** |
| WavLM-Base/Joint   | Genuine          | test       |               **0.553168** |
| WavLM-Large/Frozen | Deepfake         | training   |                   0.633382 |
| WavLM-Large/Frozen | Deepfake         | validation |                   0.643552 |
| WavLM-Large/Frozen | Deepfake         | test       |                   0.64387  |
| WavLM-Large/Frozen | Genuine          | training   |                   0.636669 |
| WavLM-Large/Frozen | Genuine          | validation |                   0.629064 |
| WavLM-Large/Frozen | Genuine          | test       |                   0.651273 |

- **WavLM-Base/Joint** front-end generally outperforms other front-emds in terms of EER for both deepfake and genuine datasets.
- **Genuine datasets** show slightly higher EER compared to deepfake datasets with WavLM.

### Deepfake Detection
| Front-End          | Triplet Mining   | Dataset    |   Deepfake Detection EER |
|:-------------------|:-----------------|:-----------|-------------------------:|
| MFCC               | Deepfake         | training   |                 0.670413 |
| MFCC               | Deepfake         | validation |                 0.663193 |
| MFCC               | Deepfake         | test       |                 0.672118 |
| MFCC               | Genuine          | training   |                 0.538294 |
| MFCC               | Genuine          | validation |                 0.544037 |
| MFCC               | Genuine          | test       |                 0.543519 |
| WavLM-Base/Frozen  | Deepfake         | training   |                 0.739727 |
| WavLM-Base/Frozen  | Deepfake         | validation |                 0.739647 |
| WavLM-Base/Frozen  | Deepfake         | test       |                 0.748872 |
| WavLM-Base/Frozen  | Genuine          | training   |                 0.541659 |
| WavLM-Base/Frozen  | Genuine          | validation |                 0.55832  |
| WavLM-Base/Frozen  | Genuine          | test       |                 0.536989 |
| WavLM-Base/Joint   | Deepfake         | training   |                 0.719853 |
| WavLM-Base/Joint   | Deepfake         | validation |                 0.723511 |
| WavLM-Base/Joint   | Deepfake         | test       |                 0.732477 |
| WavLM-Base/Joint   | Genuine          | training   |             **0.524131** |
| WavLM-Base/Joint   | Genuine          | validation |             **0.531101** |
| WavLM-Base/Joint   | Genuine          | test       |             **0.528507** |
| WavLM-Large/Frozen | Deepfake         | training   |                 0.660818 |
| WavLM-Large/Frozen | Deepfake         | validation |                 0.669061 |
| WavLM-Large/Frozen | Deepfake         | test       |                 0.674733 |
| WavLM-Large/Frozen | Genuine          | training   |                 0.565944 |
| WavLM-Large/Frozen | Genuine          | validation |                 0.572612 |
| WavLM-Large/Frozen | Genuine          | test       |                 0.575025 |

- **WavLM-Base/Joint** front-end shows better performance in detecting deepfakes with **Genuine** datasets.
- The **Genuine** dataset consistently performs better in deepfake detection across all front-ends, highlighting that genuine audio may provide more reliable features for detecting anomalies.

### Conclusion
The choice of front-end and dataset significantly impacts the performance of both speaker verification and deepfake detection systems.

- **WavLM-Base/Joint** front-end performs better in speaker verification, especially with genuine datasets.

However, the evaluation of the trained models suggest, that the datasets is not sophisticated enough, to train a model for automatic speaker verification on them. With the inability to train speaker verification, the detection of deepfakes using feature embeddings of authentic audio samples is also not feasible.

There are some possible alterations, that could result in better performance in speaker verification, as well as deepfake detection:

- [ ] Train the model on different datasets
  - [ ] VoxCeleb
  - [ ] ASVSpoof 2014
  - [ ] Full LibriSpeech Dataset
- [ ] Use the pretrained WavLM-Large/ECAPA-TDNN Model from Espnet-SPK and fine-tune to deepfake detecion using the BSI dataset
- [ ] Use pretrained Models, to fine-tune on
  - [ ] single speaker verification and
  - [ ] deepfake detection
- [ ] Change the trained model to use two embeddings at the same time. Instead of using a mono audio spur, this could use a dual audio input for audio in question and real audio. This would not be trained using triplet loss.

### Lessons Learned
There were still some errors in the model generation and EER calculation. The loss was not calculated for batches.

## Second Models LibriSpeech train-100

The main reason is to test, if Triplet Loss is able to learn speaker verification

| Front-End          | Triplet Mining   | Dataset    |   Speaker Verification EER |
|:-------------------|:-----------------|:-----------|---------------------------:|
| MFCC               | Genuine          | training   |                  0.0815375 |
| MFCC               | Genuine          | validation |                  0.0965594 |
| MFCC               | Genuine          | test       |                  0.110305  |
| WavLM-Base/Frozen  | Genuine          | training   |                  0.0957287 |
| WavLM-Base/Frozen  | Genuine          | validation |                  0.145394  |
| WavLM-Base/Frozen  | Genuine          | test       |                  0.139695  |
| WavLM-Base/Joint   | Genuine          | training   |              **0.045797**  |
| WavLM-Base/Joint   | Genuine          | validation |                  0.099889  |
| WavLM-Base/Joint   | Genuine          | test       |                  0.0961832 |
| WavLM-Large/Frozen | Genuine          | training   |                  0.0760363 |
| WavLM-Large/Frozen | Genuine          | validation |                  0.129856  |
| WavLM-Large/Frozen | Genuine          | test       |                  0.108015  |
| WavLM-Large/Joint  | Genuine          | training   |              **0.0363362** |
| WavLM-Large/Joint  | Genuine          | validation |                  0.0836108 |
| WavLM-Large/Joint  | Genuine          | test       |                  0.0675573 |

Next steps is to train on full LibriSpeech, as well as VoxCeleb (Only WavLm Base Joint)

# References

1. [ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification](https://arxiv.org/abs/2005.07143)
```latex
@inproceedings{desplanques20_interspeech,
  author={Brecht Desplanques and Jenthe Thienpondt and Kris Demuynck},
  title={{ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={3830--3834},
  doi={10.21437/Interspeech.2020-2650},
  issn={2308-457X}
}
```
2. [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)
```latex
@ARTICLE{9814838,
  author={Chen, Sanyuan and Wang, Chengyi and Chen, Zhengyang and Wu, Yu and Liu, Shujie and Chen, Zhuo and Li, Jinyu and Kanda, Naoyuki and Yoshioka, Takuya and Xiao, Xiong and Wu, Jian and Zhou, Long and Ren, Shuo and Qian, Yanmin and Qian, Yao and Wu, Jian and Zeng, Michael and Yu, Xiangzhan and Wei, Furu},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing}, 
  year={2022},
  volume={16},
  number={6},
  pages={1505-1518},
  keywords={Predictive models;Self-supervised learning;Speech processing;Speech recognition;Convolution;Benchmark testing;Self-supervised learning;speech pre-training},
  doi={10.1109/JSTSP.2022.3188113}}
```
3. [ESPnet-SPK: full pipeline speaker embedding toolkit with reproducible recipes, self-supervised front-ends, and off-the-shelf models
](https://arxiv.org/abs/2401.17230)
```latex
@misc{jung2024espnetspk,
      title={ESPnet-SPK: full pipeline speaker embedding toolkit with reproducible recipes, self-supervised front-ends, and off-the-shelf models}, 
      author={Jee-weon Jung and Wangyou Zhang and Jiatong Shi and Zakaria Aldeneh and Takuya Higuchi and Barry-John Theobald and Ahmed Hussen Abdelaziz and Shinji Watanabe},
      year={2024},
      eprint={2401.17230},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
4. [SpeechBrain: A General-Purpose Speech Toolkit
](https://arxiv.org/abs/2106.04624)
```latex
@misc{ravanelli2021speechbrain,
      title={SpeechBrain: A General-Purpose Speech Toolkit}, 
      author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
      year={2021},
      eprint={2106.04624},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
5. [SUPERB: Speech Processing Universal PERformance Benchmark](https://arxiv.org/abs/2105.01051)
```latex
@inproceedings{yang21c_interspeech,
  author={Shu-wen Yang and Po-Han Chi and Yung-Sung Chuang and Cheng-I Jeff Lai and Kushal Lakhotia and Yist Y. Lin and Andy T. Liu and Jiatong Shi and Xuankai Chang and Guan-Ting Lin and Tzu-Hsien Huang and Wei-Cheng Tseng and Ko-tik Lee and Da-Rong Liu and Zili Huang and Shuyan Dong and Shang-Wen Li and Shinji Watanabe and Abdelrahman Mohamed and Hung-yi Lee},
  title={{SUPERB: Speech Processing Universal PERformance Benchmark}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={1194--1198},
  doi={10.21437/Interspeech.2021-1775}
}
```
6. [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
```latex
@INPROCEEDINGS{7298682,
  author={Schroff, Florian and Kalenichenko, Dmitry and Philbin, James},
  booktitle={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={FaceNet: A unified embedding for face recognition and clustering}, 
  year={2015},
  volume={},
  number={},
  pages={815-823},
  keywords={Face;Face recognition;Training;Accuracy;Artificial neural networks;Standards;Principal component analysis},
  doi={10.1109/CVPR.2015.7298682}}

```
7. [Learning local feature descriptors with triplets and shallow convolutional neural networks](https://www.researchgate.net/publication/317192886_Learning_local_feature_descriptors_with_triplets_and_shallow_convolutional_neural_networks)
```latex
@inproceedings{inproceedings,
author = {Balntas, Vassileios and Riba, Edgar and Ponsa, Daniel and Mikolajczyk, Krystian},
year = {2016},
month = {01},
pages = {119.1-119.11},
title = {Learning local feature descriptors with triplets and shallow convolutional neural networks},
doi = {10.5244/C.30.119}
}
```
8. [LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech](https://arxiv.org/abs/1904.02882)
```latex
@misc{zen2019libritts,
      title={LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech}, 
      author={Heiga Zen and Viet Dang and Rob Clark and Yu Zhang and Ron J. Weiss and Ye Jia and Zhifeng Chen and Yonghui Wu},
      year={2019},
      eprint={1904.02882},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
