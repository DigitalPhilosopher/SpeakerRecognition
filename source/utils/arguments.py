
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
        "--epochs",
        type=int,
        required=False,
        default=25,
        help="Number of training epochs (default: 25)"
    )

    # Validation
    parser.add_argument(
        "--validation_rate",
        type=int,
        required=False,
        default=5,
        help="Validation rate, i.e., validate every N epochs (default: 5)"
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
    MODEL, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE = get_general_variables(args)
    DOWNSAMPLING_TRAIN, DOWNSAMPLING_TEST, DOWNSAMPLING_VALID = get_downsampling_variables(
        args)

    LEARNING_RATE = args.learning_rate
    RESTART_EPOCH = args.restart_epoch
    MARGIN = args.margin
    NORM = args.norm
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    VALIDATION_RATE = args.validation_rate
    WEIGHT_DECAY = args.weight_decay
    AMSGRAD = args.amsgrad

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

    if args.dataset == "genuine":
        FOLDER += "/Genuine"
        TAGS["Dataset"] = "Genuine"
    else:
        FOLDER += "/Deepfake"
        TAGS["Dataset"] = "Deepfake"

    return MODEL, FOLDER, TAGS, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, LEARNING_RATE, RESTART_EPOCH, MARGIN, NORM, BATCH_SIZE, EPOCHS, VALIDATION_RATE, WEIGHT_DECAY, AMSGRAD, DOWNSAMPLING_TRAIN, DOWNSAMPLING_TEST, DOWNSAMPLING_VALID


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
        "--threshold",
        type=float,
        required=False,
        default=50.0,
        help="Threshold that can not be passed for it to beeing a genuine audio"
    )

    parser = add_general_arguments(parser)

    return parser.parse_args()


def get_inference_variables(args):
    MODEL, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE = get_general_variables(args)

    REFERENCE_AUDIO = args.reference_audio
    QUESTION_AUDIO = args.audio_in_question
    THRESHOLD = args.threshold

    return MODEL, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, THRESHOLD, REFERENCE_AUDIO, QUESTION_AUDIO


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
    MODEL, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE = get_general_variables(args)
    DOWNSAMPLING_TRAIN, DOWNSAMPLING_TEST, DOWNSAMPLING_VALID = get_downsampling_variables(
        args)

    BATCH_SIZE = args.batch_size

    TRAIN = args.train
    VALID = args.valid
    TEST = args.test

    NO_GENUINE = not args.analyze_genuine
    NO_DEEPFAKE = not args.analyze_deepfake

    return MODEL, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE, DOWNSAMPLING_TRAIN, DOWNSAMPLING_TEST, DOWNSAMPLING_VALID, BATCH_SIZE, TRAIN, VALID, TEST, NO_GENUINE, NO_DEEPFAKE


def add_general_arguments(parser):
    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Which dataset to use (genuine | deepfake)"
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

    return parser


def get_general_variables(args):
    MFCCS = args.mfccs
    SAMPLE_RATE = args.sample_rate
    EMBEDDING_SIZE = args.embedding_size

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

    if args.dataset == "genuine":
        MODEL += "_Genuine"
    else:
        MODEL += "_Deepfake"

    return MODEL, MFCCS, SAMPLE_RATE, EMBEDDING_SIZE


def get_downsampling_variables(args):
    DOWNSAMPLING_TRAIN = args.downsample_train
    DOWNSAMPLING_TEST = args.downsample_test
    DOWNSAMPLING_VALID = args.downsample_valid

    return DOWNSAMPLING_TRAIN, DOWNSAMPLING_TEST, DOWNSAMPLING_VALID
