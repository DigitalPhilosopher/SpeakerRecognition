
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Training ECAPA-TDNN Model for Deepfake Speaker Verification"
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Which dataset to use (genuine | deepfake)"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        required=False,
        default=16000,
        help="Sample rate for the audio data (default: 16000)"
    )

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

    # Frontend
    parser.add_argument(
        "--mfccs",
        type=int,
        required=False,
        default=13,
        help="Number of MFCC features to extract (default: 13)"
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

    # Scheduler
    parser.add_argument(
        "--restart_epoch",
        type=int,
        required=False,
        default=5,
        help="Epoch at which to restart training (default: 5)"
    )

    return parser.parse_args()
