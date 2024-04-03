# from video_dataset_mm import  VideoFrameDataset, ImglistToTensor
from comet_ml import Experiment
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from torchvision.models.video import r3d_18
from torchvision import models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import OrderedDict


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.V = nn.Linear(input_dim, input_dim, bias=False)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(input_dim, 2)

        self.out_layer1 = nn.Linear(512, 256)
        self.out_layer2 = nn.Linear(256, 64)
        self.out_layer3 = nn.Linear(64, 2)

    def forward(self, x):
        q = self.W(x)
        attn_weights = torch.softmax(self.V(self.tanh(q)), dim=1)
        attended_x = attn_weights * x
        out = self.out_layer1(attended_x)
        out = self.out_layer2(out)
        out = self.out_layer3(out)
        return out


class SequentialEncoder(nn.Sequential):
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoderBlock, self).__init__()
        self.layers = SequentialEncoder(
            *[TransformerEncoderLayer(input_dim, num_heads, hidden_dim)
              for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)

        # Apply feed forward network
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        return x


class MultimodalTransformer_wo_JR(nn.Module):
    def __init__(self, visual_dim, audio_dim, num_heads, hidden_dim,
                 num_layers, output_format: str):
        super(MultimodalTransformer_wo_JR, self).__init__()

        assert output_format in ['FC'], output_format
        self.output_format = output_format

        self.visual_encoder = TransformerEncoderBlock(visual_dim, num_heads,
                                                      hidden_dim, num_layers)
        self.physiological_encoder = TransformerEncoderBlock(audio_dim,
                                                             num_heads,
                                                             hidden_dim,
                                                             num_layers)
        self.cross_attention_v = nn.MultiheadAttention(visual_dim, num_heads)
        self.cross_attention_p = nn.MultiheadAttention(audio_dim, num_heads)
        self.gated_attention = nn.Linear(visual_dim + audio_dim, 1)

        # unused ---------------------------------------------------------------
        # self.fc = nn.Linear(visual_dim + audio_dim, num_classes)
        #
        # self.out_layer1 = nn.Linear(1024, 512)
        # self.out_layer2 = nn.Linear(512, 256)
        # self.out_layer3 = nn.Linear(256, 64)
        # self.out_layer4 = nn.Linear(64, num_classes)
        # self.relu = nn.ReLU()
        #
        # self.layer_norm = nn.LayerNorm(1024)
        # ----------------------------------------------------------------------

        self.final_layer = nn.Linear(1024, 512)

    def forward(self, visual_features, physiological_features):
        visual_encoded = self.visual_encoder(visual_features)
        physiological_encoded = self.physiological_encoder(
            physiological_features)

        # Do all the cross-attention for visual features
        cross_attention_output_v, _ = self.cross_attention_v(
            visual_encoded.permute(1, 0, 2),
            physiological_encoded.permute(1, 0, 2),
            physiological_encoded.permute(1, 0, 2))
        cross_attention_output_v = cross_attention_output_v.permute(1, 0, 2)

        # Do all the cross-attention for physio features
        cross_attention_output_p, _ = self.cross_attention_p(
            physiological_encoded.permute(1, 0, 2),
            visual_encoded.permute(1, 0, 2), visual_encoded.permute(1, 0, 2))
        cross_attention_output_p = cross_attention_output_p.permute(1, 0, 2)

        assert self.output_format == 'FC', self.output_format

        # Concatenate the cross-attention outputs
        concat_attention = torch.cat(
            (cross_attention_output_v, cross_attention_output_p), dim=2)
        # todo: introduce attention as option, same as
        #  `MultimodalTransformer_w_JR`.
        out = self.final_layer(concat_attention)

        return out
