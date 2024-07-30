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
├── trained_models/
│ └── vit_256_6_8_30-07-2024.pth # a checkpoint of a pre-trained ViT
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

### Loading pre-trained weights
Pre-trained checkpoints files (denoted by the file type `.ckpt`) go under the folder of "trained_models". During training
checkpoints of the best validation loss model will be stored under "trained_models/model_name/." under the name "model_date.pth".
So for example if you want to load the weights of the model 'vit_256_6_8' trained in an specific date, then you should do:
```bash
python scripts/main.py --model vit_256_6_8 --pretrain y --model_checkpoint vit_256_6_8_30-07-2024.pth
```

This of course is not robust, as training the same model on the same day will remove the previous checkpoint.
The program will try to find the checkpoint of that model inside of the models' folder in 'trained_model', since 
**trained weights should fit the exact layers of the loading architecture**.