import argparse
import os
import datetime
import torch


def str2bool(v):
    """Function to interpret strings from console"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def select_dtype(data_type):
    """Function to choose the floating point type for the models and data"""
    if data_type == 'float32':
        dtype = torch.float32
        # torch.set_float32_matmul_precision("high")
    elif data_type == 'float64' or data_type == 'double':
        dtype = torch.float64
    return dtype


def load_args(logger_name=''):
    args, unknown = parser.parse_known_args()
    return args


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int)
torch.manual_seed(parser.parse_known_args()[0].seed)

if 'c703i' in os.uname()[1]:
    parser.add_argument('--device', default='cuda:0', type=str)

else:
    parser.add_argument('--device', default='cpu', type=str)

parser.add_argument('--date', default=datetime.date.today().strftime("%d-%m-%Y"), type=str)

########### Training hyperparameters ###########
parser.add_argument('--lr', default=3e-3, type=float)
parser.add_argument('--lr_scheduler', default=True, type=str2bool)
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--pretrain', default=True, type=str2bool)
parser.add_argument('--model_checkpoint', default='vit_256_6_8_30-07-2024.pth', type=str)

########### Dataset hyperparameters ###########
parser.add_argument('--img_size', default=32, type=int)
parser.add_argument('--patch_size', default=4, type=int)
parser.add_argument('--n_classes', default=10, type=int)
parser.add_argument('--data_type', default='float32', type=str)
dtype = parser.parse_known_args()[0].data_type
parser.add_argument('--dtype', default=select_dtype(dtype))

########### Model hyperparameters ###########
parser.add_argument('--model', default='vit_256_6_8', type=str)
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--vit_dim', default=256, type=int)
parser.add_argument('--vit_depth', default=6, type=int)
parser.add_argument('--n_heads', default=8, type=int)
parser.add_argument('--mlp_factor', default=4, type=int)

if __name__ == "__main__":
    args = load_args('example')
    print(args)
