import torch
from torch import nn

from transformer import Transformer

class PatchEmbedding(nn.Module):
    """
        Patch Embedding block.
    """
    def __init__(self, patch_width, patch_height, in_channels, dim_embedding):
        super(PatchEmbedding, self).__init__()

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=dim_embedding, 
                                kernel_size=(patch_height, patch_width),
                                stride=(patch_height, patch_width))

        # cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_embedding))

    def forward(self, x):
        # (batch, dim_embedding, rows, cols)
        y = self.conv2d(x)

        batch, dim_embedding, rows, cols = y.shape        
        num_patches = rows * cols

        # (batch, dim_embedding, num_patches)
        y = y.reshape(batch, dim_embedding, num_patches)
        # (batch, num_patches, dim_embedding)
        y = torch.transpose(y, 2, 1)
        
        # (batch, dim_embedding, 1)
        # cls_tokens = self.cls_token.repeat(batch, 1, 1)

        # (batch, num_patches + 1, dim_embedding)
        # y = torch.cat([cls_tokens, y], dim=1)

        return y


class SETR_PUP(nn.Module):
    def __init__(self):
        super(SETR_PUP, self).__init__()
        
        self.num_classes = 4
        self.image_width = 256
        self.image_height = 256
        self.in_channels = 3    # RGB image
        self.patch_rows = 16
        self.patch_cols = 16
        self.num_heads = 16
        self.dim_embedding = 256    # 1024

        assert self.dim_embedding % self.num_heads == 0
        assert self.image_width % self.patch_cols == 0
        assert self.image_height % self.patch_rows == 0

        self.patch_width = self.image_width // self.patch_cols
        self.patch_height = self.image_height // self.patch_rows
        self.num_patches = self.patch_cols * self.patch_rows
    
        # Image patch embedding
        self.patch_to_embedding = PatchEmbedding(patch_width=self.patch_width, patch_height=self.patch_height, in_channels=self.in_channels, dim_embedding=self.dim_embedding)
        
        # Position embedding.
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim_embedding))
        
        # x24
        transformers = []
        for i in range(8):
            transformers.append(Transformer(dim_embedding=self.dim_embedding, num_heads=self.num_heads, dropout=0.3))
        self.transformers = nn.Sequential(*transformers)

        # TODO: To check if layer norm improves performance.
        # self.layer_norm = nn.LayerNorm(self.dim_embedding)

        shrink = 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_embedding, out_channels=self.dim_embedding, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_embedding, out_channels=self.dim_embedding//shrink, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_embedding//shrink, out_channels=self.dim_embedding//shrink, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_embedding//shrink, out_channels=self.dim_embedding//shrink, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_embedding//shrink, out_channels=self.num_classes, kernel_size=1)
        )

    def forward(self, x):

        y = self.patch_to_embedding(x)
        y += self.pos_embedding

        y = self.transformers(y)
        
        # y: (batch, patches+1, dim_embedding) -> (batch, patch rows, patch cols, dim_embedding)
        y = y.reshape(-1, self.patch_rows, self.patch_cols, self.dim_embedding)
        # (batch, dim_embedding, patch rows, patch cols)
        y = y.permute(0, 3, 1, 2)

        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)

        y = self.conv5(y)
        
        return y