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

python scripts/training.py

# Folder structure

face-recognition/
│
├── data/
│ └── lfw-deepfunneled/ # Extracted LFW deep-funneled images
│
├── datasets/
│ └── lfw_dataset.py # Dataset loading and preprocessing
│
├── models/
│ └── cnn_model.py # CNN model definition
│
├── notebooks/
│ └── EDA.ipynb # For exploratory data analysis
│
├── outputs/
│ ├── model/ # Saved trained model
│ └── logs/ # Training logs (TensorBoard etc.)
│
├── scripts/
│ ├── train.py # Training pipeline
│ └── predict.py # Inference script
│
├── utils/
│ └── preprocess.py # Preprocessing functions
│
├── requirements.txt # Dependencies
└── README.md # Project overview
