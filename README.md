# Vision Transformer implementation tutorial

### Description

This is my public repo for Vision Transformer implementation. Here you will have the scripts necessary for implementing,
training and testing a vision transformer in the CIFAR10 dataset. This repo also comes with the original notebook file 
to follow the ViT pytorch implementation step by step.

## Getting Started
First make sure you have all dependencies installed. This project runs in python3.10 so maybe create a virtual environment accordingly.

### Project structure
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

### Execution
The project is ready to be run by executing:
```bash
python scripts/main.py
```
This will run a training and subsequent testing of a model defined in `config.py`. You can change directly there different parameters
like the learning rate, the number of epochs, if the model is pretrained or not...

The `config.py` file is called at the beginning of `main.py` and is also prepared to parse arguments, so for example:
```bash
python scripts/main.py --model VisionTransformer 
```
will execute a training and testing of a ViT with the architecture hyperparameters configured in `config.py`. 
On the other hand, doing
```bash
python scripts/main.py --model vit_768_12_12
```
will ignore the architecture hyperparameters in `config.py` and will apply the training and testing of vit_768_12_12 predefined
model.

As well as the model, you can also change all the other parsed arguments available in `config.py`, for example:
```bash
python scripts/main.py --model vit_768_12_12 --epochs 150 --lr 0.001 --dtype float64
```

### Precautions
