import torch
import torch.nn as nn
from torchinfo import summary  # shows a summary table of the model
from einops import rearrange, repeat  # functions for tensor dimension reordering and repeating
from einops.layers.torch import Rearrange  # same as rearrange but works as a torch layer


class InputLayer(nn.Module):
    def __init__(self, image_size, patch_size, dim, channels=3, dropout=0.):
        """
        Input layer of the transformer. This will transform the input image into
        patch embeddings.

        Args:
          image_size - (int) number of pixels in each dimension of the input image
          patch_size - (int) number of pixels in each dimension of the patches
          dim - (int) inner dimensionality of the embeddings
          channels - (int) number of input channels
          dropout - (float) dropout rate
        """
        super(InputLayer, self).__init__()

        H, W = image_size, image_size  # usually the input images are squares, but its not obligatory
        ph, pw = patch_size, patch_size
        assert H % ph == 0 and W % pw == 0  # image dimensions must be divisible by the patch size

        num_patches = (H // ph) * (W // pw)  # known in the figure as 'n'
        patch_dim = channels * ph * pw

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=ph, pw=pw),
            # 'h'/'w' number of horizontal and vertical patches. 'b' batch size
            nn.Linear(patch_dim, dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d',
                            b=b)  # Creates as many CLS tokens as samples in the batch there are
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        return x


class Attention_Block(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super(Attention_Block, self).__init__()

        dk = dim // heads  # inner head dimension. Dim and number of heads must be multiple numbers

        self.heads = heads
        self.scale = dk ** -0.5  # scale = 1 / sqrt(dk)

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)  # QKV projection
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # self.to_out = nn.Sequential(
        #     nn.Linear(dk, dim),
        #     nn.Dropout(dropout)
        # ) if dim != dk else nn.Identity()

        self.to_out = nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x)  # QKV projection
        qkv = qkv.chunk(3, dim=-1)  # split into Q, K, V

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward_Block(nn.Module):
    def __init__(self, dim, mlp_factor=4, dropout=0.):
        super(FeedForward_Block, self).__init__()

        hidden_dim = int(dim * mlp_factor)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.net(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_factor=4, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):  # We stack the transformer layers
            self.layers.append(nn.ModuleList([
                Attention_Block(dim, heads=heads, dropout=dropout),
                FeedForward_Block(dim, mlp_factor=mlp_factor, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # skip connection 1
            x = ff(x) + x  # skip connection 2
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_factor=4, dropout=0., channels=3,
                 pool='cls'):
        super(ViT, self).__init__()

        self.image_size = image_size
        self.channels = channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_factor = mlp_factor
        self.dropout = dropout

        self.input_layer = InputLayer(image_size, patch_size, dim, channels, dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_factor, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.pool = pool

    def forward(self, x):
        x = self.input_layer(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'gap' else x[:, 0]  # Here we choose, CLS token or GAP

        return self.mlp_head(x)


def VisionTransformer(args):
    model = ViT(image_size=args['img_size'],
                patch_size=args['patch_size'],
                num_classes=args['n_classes'],  # We will use CIFAR10 dataset
                dim=args['vit_dim'],
                depth=args['vit_depth'],
                heads=args['n_heads'],
                mlp_factor=args['mlp_factor'],
                dropout=args['dropout'])
    return model


# Pre-defined ViT models
# Mnemotecnic: vit_dim_depth_heads()

# ViT from tutorial
def vit_256_6_8(args):
    model = ViT(image_size=args['img_size'],
                patch_size=args['patch_size'],
                num_classes=args['n_classes'],
                dim=256,
                depth=6,
                heads=8,
                mlp_factor=4,
                dropout=args['dropout'])
    return model


# ViT Tiny:
def vit_192_12_3(args):
    model = ViT(image_size=args['img_size'],
                patch_size=args['patch_size'],
                num_classes=args['n_classes'],
                dim=192,
                depth=12,
                heads=3,
                mlp_factor=4,
                dropout=args['dropout'])
    return model


# ViT Base:
def vit_768_12_12(args):
    model = ViT(image_size=args['img_size'],
                patch_size=args['patch_size'],
                num_classes=args['n_classes'],
                dim=768,
                depth=12,
                heads=12,
                mlp_factor=4,
                dropout=args['dropout'])
    return model


# ViT Large:
def vit_1024_24_16(args):
    model = ViT(image_size=args['img_size'],
                patch_size=args['patch_size'],
                num_classes=args['n_classes'],
                dim=1024,
                depth=24,
                heads=16,
                mlp_factor=4,
                dropout=args['dropout'])
    return model


if __name__ == '__main__':
    # Create a randomized input-like tensor (resembling an image):
    x = torch.randn(1, 3, 224, 224)

    # Create a Vision Transformer
    model = ViT(image_size=224, patch_size=16, num_classes=10, dim=1024, depth=12, heads=8, mlp_factor=4)

    # Forward pass
    y = model(x)
    print(y)
    print(y.shape)
    # Summarization of the ViT created
    summary(model, input_size=x.shape, col_names=["input_size", "output_size", "num_params", "trainable"],
            row_settings=["var_names"], depth=10)
