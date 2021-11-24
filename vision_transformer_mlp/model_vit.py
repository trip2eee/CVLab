"""
    model_vit.py
    Vision Transformer for simple shape classification.
"""

import torch
from torch import nn

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
        cls_tokens = self.cls_token.repeat(batch, 1, 1)

        # (batch, num_patches + 1, dim_embedding)
        y = torch.cat([cls_tokens, y], dim=1)

        return y

class MultiHeadAttention(nn.Module):
    """
        Multi-Head Attention block
    """
    def __init__(self, dim_embedding, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()

        self.dim_embedding = dim_embedding
        self.num_heads = num_heads

        self.keys = nn.Linear(in_features=dim_embedding, out_features=dim_embedding)
        self.queries = nn.Linear(in_features=dim_embedding, out_features=dim_embedding)
        self.values = nn.Linear(in_features=dim_embedding, out_features=dim_embedding)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(in_features=dim_embedding, out_features=dim_embedding)

        assert self.dim_embedding % self.num_heads == 0

        self.d_k = self.dim_embedding // self.num_heads
        self.scale = self.d_k ** -0.5   # 1/sqrt(d_k)

    def forward(self, x):
        
        # (batch, num_patches + 1, dim_embedding)
        keys = self.keys(x)
        queries = self.queries(x)
        values = self.values(x)

        batch, num_patches1, _ = keys.shape
        # (batch, num_patches+1, num_heads, d_k)
        keys    = keys.reshape(batch, num_patches1, self.num_heads, self.d_k)
        queries = queries.reshape(batch, num_patches1, self.num_heads, self.d_k)
        values  = values.reshape(batch, num_patches1, self.num_heads, self.d_k)

        # (batch, num_heads, num_patches+1, d_k)
        keys    = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values  = values.transpose(1, 2)

        # (n, d) * (d, n) = (n, n), where n and d represent num_patches+1 and d_k, respectively.
        # (batch, num_heads, num_patches+1, num_patches+1)
        qk = torch.matmul(queries, keys.transpose(2, 3)) * self.scale

        # TODO: to apply mask

        attention = self.softmax(qk)

        # (..., num_patches+1, num_patches+1) * (..., num_patches+1, d_k) = (batch, num_heads, num_patches+1, d_k)
        qkv = torch.matmul(attention, values)

        # (batch, num_patches+1, num_heads, d_k)
        qkv = qkv.transpose(1, 2)
        qkv = qkv.reshape(batch, num_patches1, self.num_heads*self.d_k)

        y = self.projection(qkv)

        return y

class FeedForwardBlock(nn.Module):
    """
        Feed-forward block
    """
    def __init__(self, dim_embedding, expansion, dropout):
        super(FeedForwardBlock, self).__init__()

        self.linear1 = nn.Linear(in_features=dim_embedding, out_features=dim_embedding*expansion)
        self.activation1 = nn.GELU()        
        self.dropout1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(in_features=dim_embedding*expansion, out_features=dim_embedding)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        y = self.linear1(x)
        y = self.activation1(y)
        y = self.dropout1(y)

        y = self.linear2(y)
        y = self.dropout2(y)

        return y

class Transformer(nn.Module):
    """
        Transformer block
    """
    def __init__(self, dim_embedding, num_heads, expansion=4, dropout=0.0):
        super(Transformer, self).__init__()

        self.layer_norm1 = nn.LayerNorm(dim_embedding)
        self.mha = MultiHeadAttention(dim_embedding=dim_embedding, num_heads=num_heads, dropout=dropout)

        self.layer_norm2 = nn.LayerNorm(dim_embedding)

        self.feedforward = FeedForwardBlock(dim_embedding=dim_embedding, expansion=expansion, dropout=dropout)

        self.layer_norm3 = nn.LayerNorm(dim_embedding)

    def forward(self, x):
        y = self.layer_norm1(x)
        y_mha = self.mha(y)

        y = y + y_mha
        y = self.layer_norm2(y)

        y_ff = self.feedforward(y)
        y = y + y_ff

        y = self.layer_norm3(y)

        return y

class MLPHead(nn.Module):
    """
        Multiple Layer Perceptron head for classification.
    """
    def __init__(self, dim_embedding, num_classes):
        super(MLPHead, self).__init__()
        
        self.linear1 = nn.Linear(in_features=dim_embedding, out_features=dim_embedding//2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=dim_embedding//2, out_features=num_classes)

    def forward(self, x):
        y = self.linear1(x)
        y = self.relu(y)
        y = self.linear2(y)

        return y

class ShapeViT(nn.Module):
    """
        Simple shape classifier using Vision Trasnformer (ViT)
    """
    def __init__(self):
        super(ShapeViT, self).__init__()

        self.num_classes = 2
        self.image_width = 48
        self.image_height = 48
        self.in_channels = 3    # RGB image
        self.patch_rows = 3
        self.patch_cols = 3
        self.num_heads = 8
        self.dim_embedding = 64

        assert self.dim_embedding % self.num_heads == 0
        assert self.image_width % self.patch_cols == 0
        assert self.image_height % self.patch_rows == 0

        self.patch_width = self.image_width // self.patch_cols
        self.patch_height = self.image_height // self.patch_rows
        self.num_patches = self.patch_cols * self.patch_rows
    
        # Image patch embedding
        self.patch_to_embedding = PatchEmbedding(patch_width=self.patch_width, patch_height=self.patch_height, in_channels=self.in_channels, dim_embedding=self.dim_embedding)
        
        # Position embedding.
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches+1, self.dim_embedding))
        self.transformer = Transformer(dim_embedding=self.dim_embedding, num_heads=self.num_heads, dropout=0.3)
        self.mlp_head = MLPHead(dim_embedding=self.dim_embedding, num_classes=self.num_classes)

    def forward(self, x):
        y = self.patch_to_embedding(x)
        y += self.pos_embedding

        y = self.transformer(y)

        # (batch, patches+1, dim_embedding) -> (batch, dim_embedding)
        y = y.mean(dim=1)

        y = self.mlp_head(y)

        return y



