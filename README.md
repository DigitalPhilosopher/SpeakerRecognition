Cuda version 12.1 or higher needs to be installed

# Create virtual python environment 
python -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Add extraction utils
ln -s /path/to/extraction_utils source/extraction_utils

# Train Model
python source/TrainModel.py --frontend mfcc --dataset genuine --batch_size 16 --epochs 20 --validation_rate 50 --margin 0.5 --restart_epoch 50 --mfccs 80
python source/TrainModel.py --frontend mfcc --dataset deepfake --batch_size 16 --epochs 20 --validation_rate 50 --margin 0.5 --restart_epoch 50 --mfccs 80

python source/TrainModel.py --frontend wavlm_base --dataset genuine --batch_size 8 --epochs 20 --validation_rate 50 --margin 0.5 --restart_epoch 50
python source/TrainModel.py --frontend wavlm_base --dataset deepfake --batch_size 8 --epochs 20 --validation_rate 50 --margin 0.5 --restart_epoch 50

python source/TrainModel.py --frontend wavlm_base --frozen 0 --dataset genuine --batch_size 8 --epochs 20 --validation_rate 50 --margin 0.5 --restart_epoch 50
python source/TrainModel.py --frontend wavlm_base --frozen 0 --dataset deepfake --batch_size 8 --epochs 20 --validation_rate 50 --margin 0.5 --restart_epoch 50

# Show results:
mlflow ui 

## Open in browser:
http://127.0.0.1:5000/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D 