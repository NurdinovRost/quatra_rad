Quatra_Rad
==============================

### Description
Test task of Quatra_rad. Solves the problem of classifying images into 3 categories: bottle, packet, glass

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Best trained model
    │
    ├── notebooks          <- Jupyter notebooks.
    │   ├── data_analysis.ipynb      <- Raw data analysis
    │   └── inference_model.ipynb    <- Evaluation of the trained model, result and visualization
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module.
    │   ├── augmentations.py    <- Augmentations for training and test data.
    │   ├── dataset.py     <- Class Dataset for training.
    │   ├── experiment.py  <- Experiment for training model with using catalyst pipeline.
    │   ├── model.py       <- Class model for training model.
    │   ├── data           <- Scripts for handling data
    │   │   └── data_build.py          <- Script to prepare the data for training.
    │   │   └── incorrect_samples.py   <- Dict of incorrect samples.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── .gitattributes     <- Description lfs files
    ├── .gitignore         <- gitignore file.
    ├── train.yml          <- Configuration file with params for training model.
    └── 

--------

### Usage
#### Description of how to start training
1) ```git clone git@github.com:NurdinovRost/quatra_rad.git```
2) ```cd quatra_rad```
3) ```virtualenv venv```
4) ```source env/bin/activate```
5) ```pip install -r requirements.txt```
6) download raw data and move to ./data/raw/
7) ```python src/data/data_build.py```
8) ```catalyst-dl run -C train.yml```

**Note: you can change params for training in `train.yml` or start a call with the parameters you want.**
**Example:**
```
catalyst-dl run -C infer.yaml \
                --model_params/encoder_name="efficientnet_b1":str \
                --model_params/dropout_rate=0.7:float \
                --stages/data_params/batch_size="16":str \
                --stages/data_params/fold=2:int \
                --stages/optimizer_params/lr=0.0005 \
```

### Running the model on test data
To test the model you can run `./notebooks/inference_model.ipynb`