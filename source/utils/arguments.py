
import argparse


def get_training_arguments():
    parser = argparse.ArgumentParser(
        description="Training ECAPA-TDNN Model for Deepfake Speaker Verification"
    )

    parser = add_downsampling_arguments(parser)

    parser = add_general_arguments(parser)

    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=8,
        help="Batch size for training (default: 8)"
    )
    parser.add_argument(
        "--triplet_mining",
        type=str,
        required=False,
        default="random",
        help="Triplet mining strategy (random | hard)"
    )
    parser.add_argument(
        "--batch_size_test_eval",
        type=int,
        required=False,
        default=4,
        help="Batch size for training (default: 4)"
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        required=False,
        default=1,
        help="The number of gradient accumulation steps (default: 1)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=25,
        help="Number of training epochs (default: 25)"
    )

    # Pretrained_model
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default=None,
        help="The path to a pretrained model (default: None)"
    )
    # Loss
    parser.add_argument(
        "--margin",
        type=float,
        required=False,
        default=1,
        help="Margin for loss function (default: 1)"
    )
    parser.add_argument(
        "--norm",
        type=int,
        required=False,
        default=2,
        help="Normalization type (default: 2)"
    )

    # Optimizer
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        default=0.001,
        help="Learning rate for the optimizer (default: 0.001)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        required=False,
        default=0.00001,
        help="Weight decay to use for optimizing (default: 0.00001)"
    )
    parser.add_argument(
        "--amsgrad",
        type=bool,
        required=False,
        default=False,
        help="Whether to use the AMSGrad variant of Adam optimizer (default: False)"
    )

    return parser.parse_args()


def get_training_variables(args):
    MODEL, DATASET, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DEVICE = get_general_variables(
        args)
    DOWNSAMPLING_TRAIN, DOWNSAMPLING_TEST, DOWNSAMPLING_VALID, MAX_AUDIOS_TRAIN, MAX_AUDIOS_TEST, MAX_AUDIOS_VALID, MAX_AUDIO_LENGTH = get_downsampling_variables(
        args)

    LEARNING_RATE = args.learning_rate
    MODEL_PATH = args.model_path
    MARGIN = args.margin
    NORM = args.norm
    BATCH_SIZE = args.batch_size
    BATCH_SIZE_TEST_EVAL = args.batch_size_test_eval
    ACCUMULATION_STEPS = args.accumulation_steps
    EPOCHS = args.epochs
    WEIGHT_DECAY = args.weight_decay
    AMSGRAD = args.amsgrad
    TRIPLET_MINING = args.triplet_mining

    if args.frontend == "mfcc":
        FOLDER = "MFCC"
        TAGS = {
            "Frontend": "MFCC"
        }
    else:
        if args.frontend == "wavlm_base":
            FOLDER = "WavLM-Base"
            TAGS = {
                "Frontend": "WavLM-Base"
            }
        elif args.frontend == "wavlm_large":
            FOLDER = "WavLM-Large"
            TAGS = {
                "Frontend": "WavLM-Large"
            }

        if args.frozen:
            FOLDER += "/frozen"
            TAGS["Frontend-training"] = "frozen"
        else:
            FOLDER += "/joint"
            TAGS["Frontend-training"] = "joint"

    FOLDER += "/ECAPA-TDNN/Random-Triplet-Mining"
    TAGS["Model"] = "joint"
    TAGS["Triplet Mining Strategy"] = "Random Triplet Mining"

    if DATASET.split(".")[1] == "genuine":
        FOLDER += "/Genuine"
        TAGS["Dataset"] = "Genuine"
    else:
        FOLDER += "/Deepfake"
        TAGS["Dataset"] = "Deepfake"

    return (MODEL, MODEL_PATH, DATASET, FOLDER, TAGS, MFCCS, SAMPLE_RATE,
            EMBEDDING_SIZE, DEVICE, LEARNING_RATE, MARGIN, NORM, BATCH_SIZE,
            BATCH_SIZE_TEST_EVAL, ACCUMULATION_STEPS, MAX_AUDIO_LENGTH, EPOCHS,
            WEIGHT_DECAY, AMSGRAD, DOWNSAMPLING_TRAIN, DOWNSAMPLING_TEST,
            DOWNSAMPLING_VALID, TRIPLET_MINING)


def get_inference_arguments():
    parser = argparse.ArgumentParser(
        description="Inference of the ECAPA-TDNN Model for Deepfake Speaker Verification or Deepfake Detection"
    )

    # Data
    parser.add_argument(
        "--reference_audio",
        type=str,
        required=True,
        help="Genuine reference audio of speaker"
    )
    parser.add_argument(
        "--audio_in_question",
        type=str,
        required=True,
        help="Audio in question to be speaker"
    )
    parser.add_argument(
        "--audio_in_question2",
        type=str,
        required=False,
        default=None,
        help="Audio in question to be speaker"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=50.0,
        help="Threshold that can not be passed for it to beeing a genuine audio"
    )

    parser = add_general_arguments(parser)

    return parser.parse_args()


def get_inference_variables(args):
    MODEL, DATASET, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DEVICE = get_general_variables(
        args)

    REFERENCE_AUDIO = args.reference_audio
    QUESTION_AUDIO = args.audio_in_question
    QUESTION_AUDIO2 = args.audio_in_question2
    THRESHOLD = args.threshold

    return MODEL, DATASET, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DEVICE, THRESHOLD, REFERENCE_AUDIO, QUESTION_AUDIO, QUESTION_AUDIO2


def get_analytics_arguments():
    parser = argparse.ArgumentParser(
        description="Analytics of the ECAPA-TDNN Model for Deepfake Speaker Verification and Deepfake Detection"
    )

    parser.add_argument(
        "--train",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to generate analytics for the training set (default=False)"
    )
    parser.add_argument(
        "--valid",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to generate analytics for the valid set (default=False)"
    )
    parser.add_argument(
        "--test",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to generate analytics for the test set (default=True)"
    )
    parser.add_argument(
        "--valid_set",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to use a pre-defined set of tuples (from validation_sets/..) or not (default=True)"
    )

    parser = add_downsampling_arguments(parser)

    parser = add_general_arguments(parser)

    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=8,
        help="Batch size for training (default: 8)"
    )

    parser.add_argument(
        "--analyze_genuine",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to generate analytics for the genuine dataset."
    )
    parser.add_argument(
        "--analyze_deepfake",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to  generate analytics for the deepfake dataset."
    )

    return parser.parse_args()


def get_analytics_variables(args):
    MODEL, DATASET, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DEVICE = get_general_variables(
        args)
    DOWNSAMPLING_TRAIN, DOWNSAMPLING_TEST, DOWNSAMPLING_VALID, MAX_AUDIOS_TRAIN, MAX_AUDIOS_TEST, MAX_AUDIOS_VALID, MAX_AUDIO_LENGTH = get_downsampling_variables(
        args)

    BATCH_SIZE = args.batch_size

    TRAIN = args.train
    VALID = args.valid
    TEST = args.test

    NO_GENUINE = not args.analyze_genuine
    NO_DEEPFAKE = not args.analyze_deepfake
    NO_VALID_SET = not args.valid_set

    return MODEL, DATASET, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DEVICE, DOWNSAMPLING_TRAIN, DOWNSAMPLING_TEST, DOWNSAMPLING_VALID, MAX_AUDIOS_TRAIN, MAX_AUDIOS_TEST, MAX_AUDIOS_VALID, MAX_AUDIO_LENGTH, BATCH_SIZE, TRAIN, VALID, TEST, NO_GENUINE, NO_DEEPFAKE, NO_VALID_SET


def add_general_arguments(parser):
    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Which dataset to use (LibriSpeech.genuine | VoxCeleb.genuine | BSI.genuine | BSI.deepfake)"
    )

    # Frontend
    parser.add_argument(
        "--mfccs",
        type=int,
        required=False,
        default=13,
        help="Number of MFCC features to extract (default: 13)"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        required=False,
        default=16000,
        help="Sample rate for the audio data (default: 16000)"
    )
    parser.add_argument(
        "--frontend",
        type=str,
        required=True,
        help="Which frontend model to use for feature extraction (mfcc, wavlm_base, wavlm_large)"
    )
    parser.add_argument(
        "--frozen",
        type=int,
        required=False,
        default=1,
        help="Whether the frontend model is jointly trained or frozen during training (1=frozen, 0=joint)"
    )

    # Model
    parser.add_argument(
        "--embedding_size",
        type=int,
        required=False,
        default=192,
        help="Size of the embedding vector (default: 192)"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Which device to use (per default looks if cuda is available)"
    )

    return parser


def add_downsampling_arguments(parser):
    # Downsampling
    parser.add_argument(
        "--downsample_train",
        type=int,
        required=False,
        default=0,
        help="Downsample training data by a factor (default: 0 - no downsampling)"
    )
    parser.add_argument(
        "--downsample_valid",
        type=int,
        required=False,
        default=0,
        help="Downsample validation data by a factor (default: 0 - no downsampling)"
    )
    parser.add_argument(
        "--downsample_test",
        type=int,
        required=False,
        default=0,
        help="Downsample test data by a factor (default: 0 - no downsampling)"
    )

    parser.add_argument(
        "--max_audios_train",
        type=int,
        required=False,
        default=0,
        help="Set maximum audios of each speaker (default: 0 - all audios of each speaker)"
    )
    parser.add_argument(
        "--max_audios_valid",
        type=int,
        required=False,
        default=0,
        help="Set maximum audios of each speaker (default: 0 - all audios of each speaker)"
    )
    parser.add_argument(
        "--max_audios_test",
        type=int,
        required=False,
        default=0,
        help="Set maximum audios of each speaker (default: 0 - all audios of each speaker)"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        required=False,
        default=32000,
        help="Maximum length of each audio (default: 32000)"
    )

    return parser


def get_general_variables(args):
    MFCCS = args.mfccs
    SAMPLE_RATE = args.sample_rate
    EMBEDDING_SIZE = args.embedding_size
    DEVICE = args.device

    if args.frontend == "mfcc":
        MODEL = "MFCC"
    else:
        if args.frontend == "wavlm_base":
            MODEL = "WavLM-Base"
        elif args.frontend == "wavlm_large":
            MODEL = "WavLM-Large"

        if args.frozen:
            MODEL += "-frozen"
        else:
            MODEL += "-joint"

    MODEL += "_ECAPA-TDNN_Random-Triplet-Mining"

    DATASET = args.dataset
    MODEL += "_" + DATASET.split(".")[0]
    if DATASET.split(".")[1] == "genuine":
        MODEL += "-Genuine"
    else:
        MODEL += "-Deepfake"

    return MODEL, DATASET, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DEVICE


def get_downsampling_variables(args):
    DOWNSAMPLING_TRAIN = args.downsample_train
    DOWNSAMPLING_TEST = args.downsample_test
    DOWNSAMPLING_VALID = args.downsample_valid
    MAX_AUDIOS_TRAIN = args.max_audios_train
    MAX_AUDIOS_TEST = args.max_audios_test
    MAX_AUDIOS_VALID = args.max_audios_valid
    MAX_AUDIO_LENGTH = args.max_length

    return DOWNSAMPLING_TRAIN, DOWNSAMPLING_TEST, DOWNSAMPLING_VALID, MAX_AUDIOS_TRAIN, MAX_AUDIOS_TEST, MAX_AUDIOS_VALID, MAX_AUDIO_LENGTH
