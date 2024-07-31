from models.ViT import *
from models.ViT_attnVis import *
import os
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms


def get_transforms(norm_mean=[0.49139968, 0.48215841, 0.44653091], norm_std=[0.24703223, 0.24348513, 0.26158784]):
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(norm_mean, norm_std)
                                         ])

    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_std)
                                          ])
    return train_transform, test_transform


def get_datasets(datapath='data', train_prcnt=0.9):
    train_transform, test_transform = get_transforms()
    dataset = CIFAR10(root=datapath, train=True, transform=train_transform, download=True)

    train_size = int(len(dataset) * train_prcnt)
    val_size = len(dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # we apply the test transformation so the validation does not have data augmentation
    val_set.dataset.transform = test_transform

    # Loading the test set
    test_set = CIFAR10(root=datapath, train=False, transform=test_transform, download=True)

    return train_set, val_set, test_set


def get_dataloaders(args):
    # We define a set of data loaders that we can use for various purposes later.
    batch_size = args.batch_size

    train_set, val_set, test_set = get_datasets()

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,
                                   num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    return train_loader, val_loader, test_loader


def loadModel(args):
    model_name = args.model
    model = eval(f'{model_name}({vars(args)})')
    device = args.device
    if args.pretrain:

        if '_attnVis' in model_name:
            model_name = model_name.replace('_attnVis', '')

        ckpt_file = os.path.join('trained_models', model_name, args.model_checkpoint)
        print(f'Loading pre-trained model of {model_name}. Checkpoint: {ckpt_file}')

        try:
            checkpoint = torch.load(ckpt_file, map_location=torch.device(device), weights_only=True)
            model.load_state_dict(checkpoint, strict=False)
            model.to(device)
            print(f"Checkpoint {ckpt_file} loaded")
        except FileNotFoundError:
            print(f'Checkpoint file {ckpt_file} not found. The model will be loaded from scratch!')
        except RuntimeError as e:
            print(f'Error loading checkpoint: {e}. Check that the model architecture matches the checkpoint')
        except Exception as e:
            print(f'An unexpected error occurred: {e}. Model will be loaded from scratch!')

    model.to(args.dtype)
    print(f'\nSize of the architecture: {sum(p.numel() for p in model.parameters())} parameters')

    model.to(args.device)
    print("The model will be running on", args.device, "device")

    return model


if __name__ == '__main__':
    batch_size = 64

    train_set, val_set, test_set = get_datasets(datapath='../data')

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,
                                   num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    # Visualize some examples
    NUM_IMAGES = 4
    CIFAR_images = torch.stack([val_set[idx][0] for idx in range(NUM_IMAGES)], dim=0)
    img_grid = torchvision.utils.make_grid(CIFAR_images, nrow=4, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.title("Image examples of the CIFAR10 dataset")
    plt.imshow(img_grid)
    plt.axis('off')
    plt.show()
    plt.close()
