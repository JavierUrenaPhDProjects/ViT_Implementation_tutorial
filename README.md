# Vision Transformer implementation tutorial

### Description

This is my public repo for Vision Transformer implementation. Here you will have the scripts necessary for implementing,
training and testing a vision transformer in the CIFAR10 dataset. This repo also comes with the original notebook file 
to follow the ViT pytorch implementation step by step.

## Getting Started

The folder distribution of the project goes as:
```
ViT_tutorial/
│
├── config_files/ # Project configuration files
│ └── config.py # Hyperparameters of the scripts. Here you change the necessary configurations for architecture and training
│
├── notebooks/ # Jupyter notebooks
│ └── ViT_Tutorial.ipynb
│
├── models/
│ └── ViT.py # Vision Transformer architecture implementation, together with the model definitions
│
├── scripts/ # All python scripts that execute the whole project
│ └── main.py        # main script. This one is what should be executed to do the training and testing of a model
│ └── trainer.py     # definition of the 'Trainer' object that will execute the training
│ └── tester.py      # definition of the 'Tester' object that will execute the testing
│ └── dataloaders.py # Datasets and dataloaders creation script. Is configured so it loads CIFAR10
│ └── evaluation.py  # evaluation function definition. 
│
└── requirements.txt # Libraries necessary
```

The project is ready to be run by executing

`python scripts/main.py`
