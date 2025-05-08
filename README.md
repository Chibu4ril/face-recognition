# create a conda environment for this project

conda env create -f face-recognition-app.yaml

# Activate the environment

conda activate face-recognition-app

# For CPU compatibility mode run

conda remove tensorflow tensorflow-base keras
pip install tensorflow==2.16.2

# Test

python -c "import tensorflow.keras; print('tensorflow.keras is working!')"

# To begin training, run

python preprocess.py

# Folder structure

face-recognition/
│
├── data/
│ └── lfw-deepfunneled/ # Extracted LFW deep-funneled images
└── preprocess.py # Preprocessing functions
│
├── face-recognition-app.yaml # Env / Dependencies
└── README.md # Project overview
