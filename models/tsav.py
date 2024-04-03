"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import sys
from os.path import basename, dirname, abspath, join

import torch.nn as nn
import torch
from torchvision import models as tv_models


root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)

from models.video_resnet import r2plus1d_18
from models.video_resnet import VideoResNet

from models.pytorch_i3d_new import InceptionI3d
from models.I3DWSDDA import I3D_WSDDA

from utils.utils import resize_clips_for_i3d


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, input):
        return input


class VideoModel(nn.Module):
    def __init__(self, num_channels=3, init_w_R2D1: str = 'RANDOM'):
        super(VideoModel, self).__init__()

        assert init_w_R2D1 in ['RANDOM', 'KINETICS400', 'AFFWILD2',
                               'OUR_AFFWILD2'], init_w_R2D1

        pretrained = (init_w_R2D1 == 'KINETICS400')
        self.r2plus1d = r2plus1d_18(pretrained=pretrained)
        self.r2plus1d.fc = nn.Sequential(nn.Dropout(0.0),
                                         nn.Linear(in_features=self.r2plus1d.fc.in_features,
                                                   out_features=17))
        if num_channels == 4:
            new_first_layer = nn.Conv3d(in_channels=4,
                                        out_channels=self.r2plus1d.stem[0].out_channels,
                                        kernel_size=self.r2plus1d.stem[0].kernel_size,
                                        stride=self.r2plus1d.stem[0].stride,
                                        padding=self.r2plus1d.stem[0].padding,
                                        bias=False)
            # copy pre-trained weights for first 3 channels
            new_first_layer.weight.data[:, 0:3] = self.r2plus1d.stem[0].weight.data
            self.r2plus1d.stem[0] = new_first_layer
        self.modes = ["clip"]

    def forward(self, x):
        return self.r2plus1d(x)

    def flush(self):
        self.r2plus1d.flush()


class AudioModel(nn.Module):
    def __init__(self, init_w_ResNet18 :str = 'RANDOM'):
        super(AudioModel, self).__init__()

        assert init_w_ResNet18 in ['RANDOM', 'IMAGENET', 'AFFWILD2',
                                   'OUR_AFFWILD2'], init_w_ResNet18

        pretrained = (init_w_ResNet18 == 'IMAGENET')
        self.resnet = tv_models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Sequential(nn.Dropout(0.0),
                                       nn.Linear(in_features=self.resnet.fc.in_features, out_features=17))

        old_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, out_channels=self.resnet.conv1.out_channels,
                                      kernel_size=(7, 7), stride=(2, 2),
                                      padding=(3, 3), bias=False)
        if pretrained == True:
            self.resnet.conv1.weight.data.copy_(
                torch.mean(old_layer.weight.data, dim=1, keepdim=True)
            ) # mean channel

        self.modes = ["audio_features"]

    def forward(self, x):
        return self.resnet(x)


def build_i3d(init_w_I3D: str = 'RANDOM'):
    assert init_w_I3D in ['RANDOM', 'KINETICS400', 'AFFWILD2',
                          'OUR_AFFWILD2'], init_w_I3D

    i3d = InceptionI3d(400, in_channels=3)
    if init_w_I3D == 'KINETICS400':
        i3d.load_state_dict(
            torch.load(join(root_dir, 'PretrainedWeights/rgb_imagenet.pt')))

    i3d_tcn_model = I3D_WSDDA(i3d)
    # use Data parallel to load the model. The weights have been saved while
    # the model is contained in a torch.nn.DataParallel!!!!
    _i3d_tcn_model = torch.nn.DataParallel(i3d_tcn_model)

    if init_w_I3D == "AFFWILD2":
        _i3d_tcn_model.load_state_dict(
            torch.load(
                join(root_dir,
                     'PretrainedWeights/Val_model_valence_cnn_lstm_mil_64_new.t7'))["net"]
        )

    i3d_tcn_model = _i3d_tcn_model.module

    return i3d_tcn_model


class TwoStreamAuralVisualModel(nn.Module):
    def __init__(self,
                 l_vision_backbones: list,
                 l_audio_backbones: list,
                 num_channels=3,
                 init_w_ResNet18: str = 'RANDOM',
                 init_w_R2D1 : str = 'RANDOM',
                 init_w_I3D: str = 'RANDOM',
                 R2D1_ft_dim_reduce: str = 'MAX'
                 ):
        super(TwoStreamAuralVisualModel, self).__init__()

        assert isinstance(l_vision_backbones, list), type(l_vision_backbones)
        assert isinstance(l_audio_backbones, list), type(l_audio_backbones)

        assert init_w_R2D1 in ['RANDOM', 'KINETICS400', 'AFFWILD2',
                               'OUR_AFFWILD2'], init_w_R2D1
        assert init_w_ResNet18 in ['RANDOM', 'IMAGENET', 'AFFWILD2',
                                   'OUR_AFFWILD2'], init_w_ResNet18
        assert init_w_I3D in ['RANDOM', 'KINETICS400', 'OUR_AFFWILD2',
                              'AFFWILD2'], init_w_I3D

        assert R2D1_ft_dim_reduce in ['MAX', 'AVG', 'FLATTEN'], \
            R2D1_ft_dim_reduce
        self.R2D1_ft_dim_reduce = R2D1_ft_dim_reduce

        self.audio_resnet18 = None
        self.vision_i3d = None
        self.vision_r2d1 = None
        self.vision_r2d1_fc = None


        if 'R2D1' in l_vision_backbones:
            self.vision_r2d1 = VideoModel(num_channels=num_channels,
                                      init_w_R2D1=init_w_R2D1)

            if R2D1_ft_dim_reduce == 'FLATTEN':
                self.vision_r2d1_fc = nn.Linear(in_features=25088,
                                                out_features=512)
                # todo: add this layer for load, save weights of R2D1.

        if 'I3D' in l_vision_backbones:
            self.vision_i3d = build_i3d(init_w_I3D)

        if 'ResNet18' in l_audio_backbones:
            self.audio_resnet18 = AudioModel(init_w_ResNet18=init_w_ResNet18)
            self.audio_resnet18.resnet.fc = Dummy()

        self.modes = ['clip', 'audio_features']

        if self.vision_r2d1 is not None:
            assert isinstance(self.vision_r2d1.r2plus1d, VideoResNet), type(
                self.vision_r2d1.r2plus1d)
            self.vision_r2d1.r2plus1d.fc = Dummy()
            self.vision_r2d1.r2plus1d.avgpool = Dummy()

    def forward(self, audio, clip):
        #audio = x['audio_features']
        #clip = x['clip']

        ft_audio_resnet18 = None
        ft_vision_i3d = None
        ft_vision_r2d1 = None

        # Audio - ResNet18: ----------------------------------------------------
        if self.audio_resnet18 is not None:
            ft_audio_resnet18 = self.audio_resnet18(audio)
        # ----------------------------------------------------------------------

        # Vision - R2D1: -------------------------------------------------------
        if self.vision_r2d1 is not None:
            self.vision_r2d1(clip)  # output: flattened features. not useful.
            ft_vision_r2d1 = self.vision_r2d1.r2plus1d.spatial_fts
            # video ft: R2D1: seq, 512, 1, `7`, `7`
            assert ft_vision_r2d1.ndim == 5, ft_vision_r2d1.ndim
            sq, d, _, _, _ = ft_vision_r2d1.shape

            _tmp_video_ft = ft_vision_r2d1.contiguous().view(sq, d, -1)

            if self.R2D1_ft_dim_reduce == 'MAX':
                ft_vision_r2d1 = _tmp_video_ft.max(dim=2, keepdim=False)[0]

            elif self.R2D1_ft_dim_reduce == 'AVG':
                ft_vision_r2d1 = _tmp_video_ft.mean(dim=2, keepdim=False)
            elif self.R2D1_ft_dim_reduce == 'FLATTEN':
                ft_vision_r2d1 = _tmp_video_ft.contiguous().view(sq, -1)
                ft_vision_r2d1 = self.vision_r2d1_fc(ft_vision_r2d1)
            else:
                raise NotImplementedError(self.R2D1_ft_dim_reduce)
            # ft_vision_r2d1: sq, other_dims

        # ----------------------------------------------------------------------

        # Vision - I3D: --------------------------------------------------------
        if self.vision_i3d is not None:
            vis_data_for_i3d = resize_clips_for_i3d(clip)
            ft_vision_i3d = self.vision_i3d(vis_data_for_i3d)
            ft_vision_i3d, _ = torch.max(ft_vision_i3d, 1)

        return ft_audio_resnet18, ft_vision_r2d1, ft_vision_i3d

    def flush(self):
        if self.vision_r2d1 is not None:
            self.vision_r2d1.flush()