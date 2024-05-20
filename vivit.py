import torch.nn as nn
from einops import rearrange
import torch

class MLP(nn.Module):
    def __init__(self, in_shape, features, outshape, dropout_p):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_shape, features),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(features, outshape),
            nn.Dropout(dropout_p)
            )
    def forward(self, x):
        return self.layers(x)


class ViT(nn.Module):
    def __init__(self, num_heads, hidden_dimension, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_dimension)
        self.attn1 = nn.MultiheadAttention(hidden_dimension, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dimension)
        #MLP has 4x the number of number of nodes in hidden layers. See ViT paper
        self.mlp = MLP(hidden_dimension, 4 * hidden_dimension, hidden_dimension, dropout)
        
        #ViViT eq. 7
        self.qkv  = nn.Linear(hidden_dimension, 3 * hidden_dimension)

        self.hidden_dimension = hidden_dimension

    def forward(self, x):
        x = self.norm1(x)
        #print(x.shape)
        qkv = self.qkv(x)
        #print(qkv.shape)
        qkv = qkv.reshape(x.shape[0], x.shape[1], 3, self.hidden_dimension) 
        #print(qkv.shape)
        qkv = qkv.permute(2, 0, 1, 3)

        # print(qkv.shape)
        # print(qkv[0].shape, qkv[1].shape, qkv[2].shape)

        q, k, v = qkv[:3]

        sum1, a = self.attn1(q,k,v)
        #print(sum1.shape, a.shape)
        sum2 = self.mlp(self.norm2(sum1))
        return sum1 + sum2

#Tubelet embedding class
class ViTPatch(nn.Module):
    def __init__(self, channels, patch_size, embedding_dims):
        super().__init__()
        self.conv = nn.Conv3d(channels, embedding_dims, kernel_size=patch_size, stride=patch_size, groups=1)
        #self.n_patches = 196

    def forward(self, img):
        #swap the order of channels and T (frames) for conv
        #print(img.shape)
        patches = img.permute(0, 2, 1, 3, 4)

        #print("C bef:", patches.shape, img.shape)
        patches = self.conv(patches)
        #print(patches.shape)

        #print(patches.shape)
        patches = patches.flatten(2).transpose(1,2)

        #print(patches.shape)
        return patches


class VideoTransformer(nn.Module):
    def __init__(self, img_shape, patch_size, embed_dims, num_heads, layers, dropout=0.1, in_channels=3):
        super().__init__()
        num_patch = 196#(img_shape[0] * img_shape[1] * img_shape[2]) // (patch_size **3)

        print(num_patch)
        self.patch_embed = ViTPatch(in_channels, patch_size, embed_dims)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(
                torch.zeros(1, 1 + num_patch, embed_dims)
        )
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([ViT(num_heads, embed_dims, dropout) for i in range(layers)])

        self.norm = nn.LayerNorm(embed_dims)
        self.head = nn.Linear(embed_dims, 400)



    def forward(self, x):
        n_samples = x.shape[0]
        #x = x.permute(1, 0, 2, 3, 4)
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
                n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        #print(x.shape, cls_token.shape)
        x = torch.cat((cls_token, x), dim=1) 

        #print(x.shape, self.pos_embed.shape)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        print(x.shape)
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]  # just the CLS token

        x = self.head(cls_token_final)

        return x



