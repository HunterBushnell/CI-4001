conda create --name fear_sim python=3.10
conda activate fear_sim

# cd CI-BioEng-Class/Lab2B   

# Download data
pip install kagglehub
python import_data.py
# get path to data from import output print, if running in jupyterhub since too large in work/ directory

# Other preprocesing stuff?

pip install 'numpy < 2'
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install uflash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# python Lab2B.py
DATASET_ROOT="/home/fabric/.cache/kagglehub/datasets/wajahat1064/emotion-recognition-using-eeg-and-computer-games/versions/2/Dataset - Emotion Recognition data Based on EEG Signals and Computer Games/Database for Emotion Recognition System Based on EEG Signals and Various Computer Games - GAMEEMO/GAMEEMO"

# Sanity Check
python train_eeg_sweep.py --dataset-root "$DATASET_ROOT" \
  --exclude-subjects 26 \
  --channel T7 \
  --channel-list "AF3,AF4,F3,F4,F7,F8,FC5,FC6,O1,O2,P7,P8,T7,T8" \
  --sanity

# Small test
# mkdir -p data_small
ln -s "$DATASET_ROOT/(S01)" "data_small/(S01)"
# (optionally) ln -s "$DATASET_ROOT/(S02)" "data_small/(S02)"
python train_eeg_sweep.py --dataset-root "$(pwd)/data_small" --sanity

python train_eeg_sweep.py --dataset-root "$(pwd)/data_small" \
  --exclude-subjects 26 --channel T7 \
  --cv random --test-size 0.2 --epochs 2 --batch-size 128 \
  --depths 1 --base-chs 8 --outputs outputs_smoke

# Run Sweep
python train_eeg_sweep.py --dataset-root "$DATASET_ROOT" \
  --exclude-subjects 26 \
  --channel T7 \
  --test-size 0.2 --cv random --epochs 30 --batch-size 32 \
  --depths 1,2,3,2,3 --base-chs 8,8,8,16,16 \
  --outputs outputs_full


# (Optional) Subject-wise evaluation (LOGO):
python train_eeg_sweep.py \
  --dataset-root "$DATASET_ROOT" \
  --cv logo --epochs 10 --save-folds \
  --depths 1,2,3,2,3 --base-chs 8,8,8,16,16 \
  --outputs outputs_logo
