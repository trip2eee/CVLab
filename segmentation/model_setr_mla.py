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


class SETR_MLA(nn.Module):
    def __init__(self):
        super(SETR_MLA, self).__init__()
        
        self.num_classes = 4
        self.image_width = 256
        self.image_height = 256
        self.in_channels = 3    # RGB image
        self.patch_rows = 16
        self.patch_cols = 16
        self.num_heads = 16
        self.dim_embedding = 64    # 1024

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
        
        self.transformers1 = self.__create_transformer_block(2)  # x6
        self.transformers2 = self.__create_transformer_block(2)  # x6
        self.transformers3 = self.__create_transformer_block(2)  # x6
        self.transformers4 = self.__create_transformer_block(2)  # x6

        # TODO: To check if layer norm improves performance.
        # self.layer_norm = nn.LayerNorm(self.dim_embedding)

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_embedding, out_channels=self.dim_embedding//2, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dim_embedding//2, out_channels=self.dim_embedding, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dim_embedding, out_channels=self.dim_embedding//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_embedding, out_channels=self.dim_embedding//2, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dim_embedding//2, out_channels=self.dim_embedding, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dim_embedding, out_channels=self.dim_embedding//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_embedding, out_channels=self.dim_embedding//2, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dim_embedding//2, out_channels=self.dim_embedding, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dim_embedding, out_channels=self.dim_embedding//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

        self.conv2_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_embedding, out_channels=self.dim_embedding//2, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dim_embedding//2, out_channels=self.dim_embedding, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dim_embedding, out_channels=self.dim_embedding//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )


        
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_embedding//2, out_channels=self.dim_embedding//2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_embedding//2, out_channels=self.dim_embedding//2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_embedding//2, out_channels=self.dim_embedding//2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv3_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_embedding//2, out_channels=self.dim_embedding//2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_embedding*2, out_channels=self.dim_embedding*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(in_channels=self.dim_embedding*2, out_channels=self.num_classes, kernel_size=1)
        )

    def __create_transformer_block(self, num_transformers, dropout=0.3):
        transformers = []
        for i in range(num_transformers):
            transformers.append(Transformer(dim_embedding=self.dim_embedding, num_heads=self.num_heads, dropout=dropout))
        return nn.Sequential(*transformers)

    def forward(self, x):

        y = self.patch_to_embedding(x)
        y += self.pos_embedding

        y1 = self.transformers1(y)
        y2 = self.transformers2(y1)
        y3 = self.transformers3(y2)
        y4 = self.transformers4(y3)

        # y = self.layer_norm(y)
        
        # y: (batch, patches+1, dim_embedding) 
        # -> (batch, patch rows, patch cols, dim_embedding) 
        # -> (batch, dim_embedding, patch rows, patch cols)
        y1 = y1.reshape(-1, self.patch_rows, self.patch_cols, self.dim_embedding)        
        y1 = y1.permute(0, 3, 1, 2)

        y2 = y2.reshape(-1, self.patch_rows, self.patch_cols, self.dim_embedding)        
        y2 = y2.permute(0, 3, 1, 2)

        y3 = y3.reshape(-1, self.patch_rows, self.patch_cols, self.dim_embedding)        
        y3 = y3.permute(0, 3, 1, 2)

        y4 = y4.reshape(-1, self.patch_rows, self.patch_cols, self.dim_embedding)        
        y4 = y4.permute(0, 3, 1, 2)

        y_conv1 = self.conv2_1(y1)
        y_conv2 = self.conv2_2(y2)
        y_conv3 = self.conv2_3(y3)
        y_conv4 = self.conv2_4(y4)

        sum_y4 = y_conv4        
        sum_y3 = torch.add(sum_y4, y_conv3)
        sum_y2 = torch.add(sum_y3, y_conv2)
        sum_y1 = torch.add(sum_y2, y_conv1)

        y1 = self.conv3_1(sum_y1)
        y2 = self.conv3_1(sum_y2)
        y3 = self.conv3_1(sum_y3)
        y4 = self.conv3_1(sum_y4)

        y = torch.cat([y1, y2, y3, y4], dim=1)

        y = self.conv5(y)
        
        return y