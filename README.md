
<div align="center">

# Senescence Reversion in Plant Images Using Perception and Unpaired Data
### Esteban A. Esquivel-Barboza ~ Jose Carranza-Rojas 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>


</div>

## Description
This repository contains the supporting code and experiments for the paper "Senescence Reversion in Plant Images Using Perception and Unpaired Data".

## How to run
Install dependencies
```yaml
# clone project
git clone https://github.com/eadan97/lat-6765
cd lat-6765

# [OPTIONAL] create conda environment
conda env create -f conda_env_gpu.yaml -n myenv
conda activate myenv

# install requirements
pip install -r requirements.txt
```

Train model with default configuration
```yaml
# default
python run.py

# train on CPU
python run.py trainer.gpus=0

# train on GPU
python run.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)
```yaml
python run.py experiment=experiment_name
```

You can override any parameter from command line like this
```yaml
python run.py trainer.max_epochs=20 datamodule.batch_size=64
```

<br>


## Project Structure
The directory structure of new project looks like this:
```
├── configs                 <- Hydra configuration files
│   ├── callbacks               <- Callbacks configs
│   ├── datamodule              <- Datamodule configs
│   ├── experiment              <- Experiment configs
│   ├── hparams_search          <- Hyperparameter search configs
│   ├── hydra                   <- Hydra related configs
│   ├── logger                  <- Logger configs
│   ├── model                   <- Model configs
│   ├── trainer                 <- Trainer configs
│   │
│   └── config.yaml             <- Main project configuration file
│
├── data                    <- Project data
│
├── logs                    <- Logs generated by Hydra and PyTorch Lightning loggers
│
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for ordering),
│                              the creator's initials, and a short `-` delimited description, e.g.
│                              `1.0-jqp-initial-data-exploration.ipynb`.
│
├── tests                   <- Tests of any kind
│   ├── smoke
│   └── unit
│
├── src
│   ├── callbacks               <- Lightning callbacks
│   ├── datamodules             <- Lightning datamodules
│   ├── models                  <- Lightning models
│   ├── utils                   <- Utility scripts
│   │
│   └── train.py                <- Training pipeline
│
├── run.py                  <- Run any pipeline with chosen experiment configuration
│
├── .env.example            <- Template of the file for storing private environment variables
├── .gitignore              <- List of files/folders ignored by git
├── .pre-commit-config.yaml <- Configuration of automatic code formatting
├── conda_env_gpu.yaml      <- File for installing conda environment
├── Dockerfile              <- File for building docker container
├── requirements.txt        <- File for installing python dependencies
├── setup.cgf               <- Configurations of linters and pytest
├── LICENSE
└── README.md
```
<br>
