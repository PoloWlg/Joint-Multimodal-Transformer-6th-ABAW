from __future__ import absolute_import
from __future__ import division

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from .mm_multi_transformers import MultimodalTransformer_w_JR
from .mm_multi_transformers import FeatureConcatFC
from .mm_transformers import MultimodalTransformer_wo_JR


__all__ = ['Two_transformers', 'SingleBackbonePretrainer']


class Two_transformers(nn.Module):
    def __init__(self,
                 v_dropout: float,
                 a_dropout: float,
                 num_heads: int,
                 num_layers: int,
                 joint_modalities: str,
                 output_format: str = 'FC',
                 vision_in_ft: int = 512
                 ):
        super(Two_transformers, self).__init__()

        assert isinstance(v_dropout, float), type(v_dropout)
        assert 0.0 <= v_dropout < 1., v_dropout
        self.v_dropout = v_dropout

        assert isinstance(a_dropout, float), type(a_dropout)
        assert 0.0 <= a_dropout < 1., a_dropout
        self.a_dropout = a_dropout

        assert isinstance(num_heads, int), type(num_heads)
        assert num_heads > 0, num_heads
        self.num_heads = num_heads

        assert isinstance(num_layers, int), type(num_layers)
        assert num_layers > 0, num_layers
        self.num_layers = num_layers

        assert isinstance(joint_modalities, str), type(joint_modalities)
        assert joint_modalities in ['NONE', 'TRANSFORMER', 'FC'], \
            joint_modalities
        self.joint_modalities = joint_modalities

        assert isinstance(vision_in_ft, int), type(vision_in_ft)
        assert vision_in_ft > 0, vision_in_ft
        self.vision_in_ft = vision_in_ft

        self.linear = None
        if vision_in_ft != 512:
            self.linear = nn.Linear(vision_in_ft, 512)

        assert output_format in ['FC', 'SELF_ATTEN'], output_format
        self.output_format = output_format

        if joint_modalities == 'TRANSFORMER':

            assert output_format in ['FC', 'SELF_ATTEN'], output_format

            self.mm_transformer = MultimodalTransformer_w_JR(
                visual_dim=512,
                audio_dim=512,
                num_heads=num_heads,
                hidden_dim=512,
                num_layers=num_layers,
                output_format=output_format
            )

            if output_format == 'FC':
                dim = 1024
            elif output_format == 'SELF_ATTEN':
                dim = 512
            else:
                raise NotImplementedError(output_format)

        elif joint_modalities == 'FC':  # Concat with FC, no JR.

            self.mm_transformer = FeatureConcatFC(512, 512)
            dim = 512

        elif joint_modalities == 'NONE':  # Vanilla, no JR

            # todo: include self-attention.
            assert output_format in ['FC'], output_format

            self.mm_transformer = MultimodalTransformer_wo_JR(
                visual_dim=512,
                audio_dim=512,
                num_heads=num_heads,
                hidden_dim=512,
                num_layers=num_layers,
                output_format=output_format
            )
            dim = 512

        else:
            raise NotImplementedError(joint_modalities)

        self.vregressor = nn.Sequential(nn.Linear(dim, 128),
                                        nn.ReLU(inplace=False),
                                        nn.Dropout(v_dropout),
                                        nn.Linear(128, 1)
                                        )

        self.aregressor = nn.Sequential(nn.Linear(dim, 128),
                                        nn.ReLU(inplace=False),
                                        nn.Dropout(a_dropout),
                                        nn.Linear(128, 1)
                                        )

    def forward(self, f1_norm, f2_norm):

        video = F.normalize(f2_norm, dim=-1)  # bsz, seq, ft
        audio = F.normalize(f1_norm, dim=-1)
        if self.linear is not None:
            video = self.linear(video)  # reduce to 512.

        audiovisualfeatures = self.mm_transformer(video, audio)  # bsz, seq, ft

        vouts = self.vregressor(audiovisualfeatures) 
        aouts = self.aregressor(audiovisualfeatures) 

        return vouts.squeeze(2), aouts.squeeze(2)


class SingleBackbonePretrainer(nn.Module):
    def __init__(self,
                 v_dropout: float,
                 a_dropout: float,):
        super(SingleBackbonePretrainer, self).__init__()

        assert isinstance(v_dropout, float), type(v_dropout)
        assert 0.0 <= v_dropout < 1., v_dropout
        self.v_dropout = v_dropout

        assert isinstance(a_dropout, float), type(a_dropout)
        assert 0.0 <= a_dropout < 1., a_dropout
        self.a_dropout = a_dropout

        dim = 512

        self.regressor = nn.Sequential(nn.Linear(dim, 128),
                                       nn.ReLU(inplace=False),
                                       nn.Dropout(a_dropout),
                                       nn.Linear(128, 2)
                                       )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 3, x.ndim

        out = self.regressor(x)  # bsz, seq, 2
        vouts = out[:, :, 0]  # bsz, seq
        aouts = out[:, :, 1]  # bsz, seq
        assert vouts.ndim == 2, vouts.ndim
        assert aouts.ndim == 2, aouts.ndim

        return vouts, aouts
