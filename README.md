# GenIAS: Generator for Instantiating Anomalies in time Series

## Overview
GenIAS is designed to produce diverse and realistic synthetic anomalies for TSAD tasks. This repository provides tools for training, evaluating, and visualizing GenIAS and other anomaly generation models.

## Installation
### Requirements
Ensure you have Python installed along with the necessary dependencies:
```sh
pip install -r requirements.txt
```

## Usage
### Training a Model
Run the training script to train the model for specific dataset (eg. MSL):
```sh
python genias_train.py --dataset 'msl' --epochs 1000 --learning_rate 0.0001 --window_size 200 --batch_size 100 --hid_size 200 --latent_dim 100
```

### Inference / Visualization
Use the provided Jupyter notebook to perform inference:
```sh
jupyter notebook genias_inference.ipynb
```

### Evaluation
Generate anomalous samples by different models (GenIAS, CARLA, COUTA, CutAddPaste, and NCAD):
```sh
python evaluation/get_representations.py
```
Evaluate the generated anomalies using:
```sh
python evaluation/get_gen_metrics.py
```

### Comparison Visualization
Use the provided Jupyter notebook to perform inference:
```sh
jupyter notebook /evaluation/gen_metrics_vis.ipynb
```


## Directory Structure
```
├── genias_train.py         # Training script
├── genias_inference.ipynb  # Inference notebook
├── losses.py               # Loss functions
├── models.py               # Model definitions
├── train_utils.py          # Training utilities
├── evaluation/             # Evaluation scripts and notebooks
│   ├── deepsvdd_adv.py
│   ├── gen_metrics.py
│   ├── gen_metrics_vis.ipynb
│   ├── get_gen_metrics.py
│   ├── get_representations.py
│   ├── injection_models.py
│   ├── representations.py
│   ├── ts_loader.py
```

## License
This project is licensed under the MIT License.

