import torch
from torch import nn, einsum
from einops import rearrange

class PhonemeEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim = 192,
        blocks = 6,
        attn_hidden_dim = 192,
        attn_heads = 2,
        kernel_size = 3,
        filter_size = 768,
        dropout = 0.1
    ):
        super().__init__()

    def forward(self, x):
        return x

class Durator(nn.Module):
    def __init__(
        self,
        *,
        duration_kernel_size = 3,
        duration_filter_size = 192,
        dropout = 0.5,
        upsampling_kernel_size = 3,
        upsampling_filter_size = 8,
    ):
        super().__init__()

    def forward(self, x):
        return x

class PriorPosterior(nn.Module):
    def __init__(
        self,
        *,
        layers = 4,
        dilation = 1,
        kernel_size = 5,
        filter_size = 192,
        wavenet_layers = 4,
    ):
        super().__init__()

    def forward(self, x):
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        *,
        conv_blocks = 4,
        conv_block_hiddens = (256, 128, 64, 32),
        upsampling_ratio = (8, 8, 2, 2),
        conv_layers = 3,
        conv_layer_kernel_size = (3, 7, 11),
        conv_dilation = (1, 3, 5),
        mem_bank_size = 1000,
        mem_bank_dim = 192,
        mem_bank_attn_heads = 2
    ):
        super().__init__()

    def forward(self, x):
        return x

class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        *,
        wavenet_layers = 16,
        dilation = 1,
        kernel_size = 5,
        filter_size = 192,
    ):
        super().__init__()

    def forward(self, x):
        return x

class Discriminator(nn.Module):
    def __init__(
        self,
        *,
        periods = (1, 2, 3, 5, 7, 11)
    ):
        super().__init__()

    def forward(self, x):
        return x

class NaturalSpeech(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
