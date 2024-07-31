import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config_files.config import load_args
from dataloaders import get_datasets, loadModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import random
import cv2
import numpy as np


def get_random_image(set='test', batch_size=1):
    train_set, val_set, test_set = get_datasets()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,
                              num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)

    if set == 'test':
        images, _ = next(iter(test_loader))
    elif set == 'train':
        images, _ = next(iter(train_loader))
    else:
        images, _ = next(iter(val_loader))
    return images


def get_attention_map(model, img, get_mask=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = img.to(device)
    img_size = (img.shape[-2], img.shape[-1])

    # Perform the forward pass
    with torch.no_grad():
        _, att_mat = model(img)

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    attn_maps_layers = []
    for joint_attn in joint_attentions:
        v = joint_attn
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        if get_mask:
            result = cv2.resize(mask / mask.max(), img_size)
        else:
            mask = cv2.resize(mask / mask.max(), img_size)[..., np.newaxis]
            result = (mask * img).astype("uint8")
        attn_maps_layers.append(result)

    # v = joint_attentions[-1]
    # grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    # mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    # if get_mask:
    #     result = cv2.resize(mask / mask.max(), img_size)
    # else:
    #     mask = cv2.resize(mask / mask.max(), img_size)[..., np.newaxis]
    #     result = (mask * img).astype("uint8")
    return attn_maps_layers


def plot_attention_map(original_img, att_maps):
    total_images = 1 + len(att_maps)

    # Create a figure with one row and multiple columns
    fig, axes = plt.subplots(nrows=1, ncols=total_images, figsize=(16, 8))

    # Display the original image in the first column
    axes[0].set_title('Original')
    axes[0].imshow(original_img.squeeze(0).permute(1, 2, 0))
    axes[0].axis('off')  # Optionally turn off axis

    # Display each attention map in the subsequent columns
    for i, att_map in enumerate(att_maps):
        axes[i + 1].set_title(f'Attention Map {i + 1}')
        axes[i + 1].imshow(att_map)
        axes[i + 1].axis('off')  # Optionally turn off axis

    plt.tight_layout()
    plt.show()

    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    # ax1.set_title('Original')
    # ax2.set_title('Attention Map Last Layer')
    # _ = ax1.imshow(original_img.squeeze(0).permute(1, 2, 0))
    # _ = ax2.imshow(att_maps[1])
    # plt.show()


if __name__ == '__main__':
    os.chdir('..')
    args = load_args("Attention maps visualization")
    torch.manual_seed(random.randint(0, 1000))
    args.model = args.model + '_attnVis'
    model = loadModel(args)
    images = get_random_image(set='eval')
    layer_idx = 3  # Configurable parameter for layer
    head_idx = 0  # Configurable parameter for head
    # visualize_attention_maps(model, images, layer_idx, head_idx)
    att_map = get_attention_map(model, images)
    plot_attention_map(images, att_map)
