#################################################
# MiniLab 1A
# ################################################

# conda create --name fear_sim python=3.10
# conda activate fear_sim

# pip install neuron
# pip install bmtk
# pip install uflash


# cd fear_simulation/components/mechanisms
# nrnivmodl .
# cd ../..


# python build_network.py
# python update_configs.py

# python run_bionet.py config.json

# python check_output.py


#################################################
# MiniLab 1B
# ################################################

conda create --name fear_sim python=3.10
conda activate fear_sim

pip install 'numpy < 2'
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install uflash

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python eeg_emotion_classification.py


#################################################
# MiniLab 1B
# ################################################
