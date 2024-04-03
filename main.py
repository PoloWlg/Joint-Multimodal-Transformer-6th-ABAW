import os
import yaml

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import copy
import datetime as dt
from collections import OrderedDict

import torch.nn as nn
import torch.nn.parallel
import torch.optim
from train import train
from val import validate
from test import Test
from models.two_transformers import Two_transformers
from models.two_transformers import SingleBackbonePretrainer
from models.tsav import TwoStreamAuralVisualModel
import sys
from os.path import dirname, abspath, join
from datasets.dataset_new import ImageList
from datasets.dataset_val import ImageList_val
from datasets.dataset_test import ImageList_test
from losses.loss import CCCLoss

from parseit import parse_input
from parseit import Dict2Obj
from instantiator import get_optimizer_for_params
import dllogger as DLLogger
from tools import fmsg
from tools import plot_tracker
from tools import state_dict_to_cpu
from tools import state_dict_to_gpu
from tools import MyDataParallel
from reproducibility import set_seed
from models.fc_layer import FcLayer
from padSequence import TrainPadSequence, ValPadSequence, TestPadSequence
from models.intra_modal_transformer_fusion import Intra_modal_transformer_fusion

_TRAIN = "train"
_VALID = "valid"
_VALENCE = "valence"
_AROUSAL = "arousal"
_AVG = "avg_valence_arousal"


def get_state_dict(m) -> dict:
    if isinstance(m, MyDataParallel):
        return m.module.state_dict()
    else:
        return m.state_dict()


def load_clean_weights(w_path, map_location):
    state_dict = torch.load(w_path, map_location='cpu')
    dp = False
    for k, v in state_dict.items():
        _dp = k.startswith('module.')
        if _dp:
            dp = True
            break
    if dp:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        state_dict = new_state_dict

    state_dict = state_dict_to_gpu(state_dict, device=map_location)

    return state_dict


def deepcopy_state_dict(model) -> dict:
    if model is None:
        return {}

    out = {
        "all": copy.deepcopy(get_state_dict(model))
    }

    # split:
    if model.audio_resnet18 is not None:
        audio_resnet18_state = state_dict_to_cpu(
            get_state_dict(model.audio_resnet18))
        out['audio_resnet18'] = audio_resnet18_state

    if model.vision_r2d1 is not None:
        vision_r2d1_state = state_dict_to_cpu(
            get_state_dict(model.vision_r2d1))
        out['vision_r2d1'] = vision_r2d1_state

        if model.vision_r2d1_fc is not None:
            vision_r2d1_fc_state = state_dict_to_cpu(
                get_state_dict(model.vision_r2d1_fc))
            out['vision_r2d1_fc'] = vision_r2d1_fc_state

    if model.vision_i3d is not None:
        vision_i3d_state = state_dict_to_cpu(
            get_state_dict(model.vision_i3d))
        out['vision_i3d'] = vision_i3d_state

    return out


def dump_models_into_disk(model_dict,
                          fusion_model,
                          backbone_pretrainer,
                          fc_layer_for_audio_concat,
                          transformer_audio_modality_fusion,
                          fc_layer_for_video_concat,
                          transformer_visio_modality_fusion,
                          epoch,
                          path):
    DLLogger.log(f'Storing best models on disk @epoch: {epoch}')

    if fusion_model is not None:
        fusion_model_state = state_dict_to_cpu(get_state_dict(fusion_model))
        torch.save(fusion_model_state, join(path, "fusion_w.pt"))

    if backbone_pretrainer is not None:
        backbone_pretrainer_state = state_dict_to_cpu(
            get_state_dict(backbone_pretrainer))
        torch.save(backbone_pretrainer_state,
                   join(path, "backbone_pretrainer_w.pt"))

    if model_dict != {}:
        model_state = state_dict_to_cpu(model_dict['all'])
        torch.save(model_state, join(path, "all_backbones.pt"))

        # split:
        if "audio_resnet18" in model_dict:
            audio_resnet18_state = state_dict_to_cpu(
                model_dict['audio_resnet18'])
            torch.save(audio_resnet18_state, join(path, "audio_resnet18.pt"))

        if "vision_r2d1" in model_dict:
            vision_r2d1_state = state_dict_to_cpu(model_dict['vision_r2d1'])
            torch.save(vision_r2d1_state, join(path, "vision_r2d1.pt"))

            if 'vision_r2d1_fc' in model_dict:
                vision_r2d1_fc_state = state_dict_to_cpu(
                    model_dict['vision_r2d1_fc'])
                torch.save(vision_r2d1_fc_state, join(path,
                                                      "vision_r2d1_fc.pt"))

        if 'vision_i3d' in model_dict:
            vision_i3d_state = state_dict_to_cpu(model_dict['vision_i3d'])
            torch.save(vision_i3d_state, join(path, "vision_i3d.pt"))


    if fc_layer_for_audio_concat is not None:
        fc_layer_for_audio_concat_state = state_dict_to_cpu(
            get_state_dict(fc_layer_for_audio_concat))
        torch.save(fc_layer_for_audio_concat_state,
            join(path, "fc_layer_for_audio_concat.pt"),
        )

    if transformer_audio_modality_fusion is not None:
        transformer_audio_modality_fusion_state = state_dict_to_cpu(
            get_state_dict(transformer_audio_modality_fusion)
        )
        torch.save(transformer_audio_modality_fusion_state,
                   join(path, "transformer_audio_modality_fusion.pt"))

    if fc_layer_for_video_concat is not None:
        fc_layer_for_video_concat_state = state_dict_to_cpu(
            get_state_dict(fc_layer_for_video_concat))
        torch.save(fc_layer_for_video_concat_state,
            join(path, "fc_layer_for_video_concat.pt"),
        )

    if transformer_visio_modality_fusion is not None:
        transformer_visio_modality_fusion_state = state_dict_to_cpu(
            get_state_dict(transformer_visio_modality_fusion)
        )
        torch.save(transformer_visio_modality_fusion_state,
                   join(path, 'transformer_visio_modality_fusion.pt'))


def main():
    root_dir = dirname(abspath(__file__))
    default_config_file = join(root_dir, "config_file.json")

    args, mode, eval_config = parse_input(config_file=default_config_file)

    # NEW: set valid/test hyper-parameters to be the same as train
    for k in ["val_params", "test_params"]:
        args[k]["seq_length"] = args["train_params"]["seq_length"]
        args[k]["subseq_length"] = args["train_params"]["subseq_length"]
        args[k]["stride"] = args["train_params"]["stride"]
        args[k]["dilation"] = args["train_params"]["dilation"]

    # overwrite.
    with open(join(args["outd"], "config.yml"), "w") as fyaml:
        yaml.dump(args, fyaml)

    path = join(args['outd'], "SavedWeights")
    os.makedirs(path, exist_ok=True)

    args = Dict2Obj(args)

    default_seed = args.SEED

    # Backbone
    model = TwoStreamAuralVisualModel(
        l_vision_backbones=args.model_params['l_vision_backbones'],
        l_audio_backbones=args.model_params['l_audio_backbones'],
        num_channels=4,
        init_w_ResNet18=args.model_params['init_w_ResNet18'],
        init_w_R2D1=args.model_params['init_w_R2D1'],
        init_w_I3D=args.model_params['init_w_I3D'],
        R2D1_ft_dim_reduce=args.model_params['R2D1_ft_dim_reduce']
    )

    cmd = 'R2D1' in args.model_params['l_vision_backbones']
    cmd &= (args.model_params['init_w_R2D1'] == 'AFFWILD2')

    if cmd:
        path_vision = join(root_dir, "PretrainedWeights/vision_TSAV_Sub4_544k.pt")
        vision_w = load_clean_weights(
            path_vision,
            map_location=torch.device('cpu')
        )
        model.vision_r2d1.load_state_dict(vision_w, strict=True)

        DLLogger.log(f"Loaded weights for vision backbone "
                     f"[R2D1, AFFWILD2]:"
                     f" {path_vision}")

    cmd = 'ResNet18' in args.model_params['l_audio_backbones']
    cmd &= (args.model_params['init_w_ResNet18'] == 'AFFWILD2')
    if cmd:
        path_audio = join(root_dir, "PretrainedWeights/audio_TSAV_Sub4_544k.pt")
        audio_w = load_clean_weights(
            path_audio,
            map_location=torch.device('cpu')
        )
        model.audio_resnet18.load_state_dict(audio_w, strict=True)
        DLLogger.log(f"Loaded weights for audio backbone "
                     f"[ResNet18, AFFWILD2]:"
                     f" {path_audio}")

    cmd = 'ResNet18' in args.model_params['l_audio_backbones']
    cmd &= (args.model_params['init_w_ResNet18'] == 'OUR_AFFWILD2')
    if cmd:
        path_audio = join(root_dir,
                          "PretrainedWeights/ResNet18_OUR_AffWild2/SavedWeights/audio_resnet18.pt")
        audio_w = load_clean_weights(
            path_audio,
            map_location=torch.device('cpu')
        )
        model.audio_resnet18.load_state_dict(audio_w, strict=True)
        DLLogger.log(f"Loaded weights for audio backbone "
                     f"[ResNet18, OUR AFFWILD2]:"
                     f" {path_audio}")

    if 'R2D1' in args.model_params['l_vision_backbones']:
        new_first_layer = nn.Conv3d(
            in_channels=3,
            out_channels=model.vision_r2d1.r2plus1d.stem[0].out_channels,
            kernel_size=model.vision_r2d1.r2plus1d.stem[0].kernel_size,
            stride=model.vision_r2d1.r2plus1d.stem[0].stride,
            padding=model.vision_r2d1.r2plus1d.stem[0].padding,
            bias=False,
        )

        new_first_layer.weight.data = model.vision_r2d1.r2plus1d.stem[0].weight.data[:, 0:3]
        model.vision_r2d1.r2plus1d.stem[0] = new_first_layer

    cmd = 'R2D1' in args.model_params['l_vision_backbones']
    cmd &= (args.model_params['init_w_R2D1'] == 'OUR_AFFWILD2')

    if cmd:
        # this model was pretrained with MAX pooling.
        assert args.model_params['R2D1_ft_dim_reduce'] == 'MAX', args.model_params['R2D1_ft_dim_reduce']
        path_vision = join(root_dir,
                           "PretrainedWeights/R2D1_OUR_AffWild2/SavedWeights/vision_r2d1.pt")
        vision_w = load_clean_weights(
            path_vision,
            map_location=torch.device('cpu')
        )
        model.vision_r2d1.load_state_dict(vision_w, strict=True)

        DLLogger.log(f"Loaded weights for vision backbone "
                     f"[R2D1, OUR AFFWILD2]:"
                     f" {path_vision}")

    cmd = 'I3D' in args.model_params['l_vision_backbones']
    cmd &= (args.model_params['init_w_I3D'] == 'OUR_AFFWILD2')

    if cmd:
        path_vision = join(root_dir,
                           "PretrainedWeights/I3D_OUR_AffWild2/SavedWeights/vision_i3d.pt")
        vision_w = load_clean_weights(
            path_vision,
            map_location=torch.device('cpu')
        )
        model.vision_i3d.load_state_dict(vision_w, strict=True)

        DLLogger.log(f"Loaded weights for vision backbone "
                     f"[I3D, OUR AFFWILD2]:"
                     f" {path_vision}")

    if torch.cuda.device_count() > 1:
        DLLogger.log(f"Using multi-gpus: {torch.cuda.device_count()} GPUs.")
        model = MyDataParallel(model)

    model = model.cuda()

    fc_layer_for_audio_concat = None
    transformer_audio_modality_fusion = None

    if len(args.model_params["l_audio_backbones"]) == 2:
        assert 'wavLM' in args.model_params["l_audio_backbones"]
        assert 'ResNet18' in args.model_params["l_audio_backbones"]

        if args.model_params["intra_modal_fusion"] == "feat_concat_fc":
            fc_layer_for_audio_concat = FcLayer(512 + 768, 512)

            if torch.cuda.device_count() > 1:
                DLLogger.log(
                    f"Using multi-gpus: {torch.cuda.device_count()} GPUs.")
                fc_layer_for_audio_concat = MyDataParallel(
                    fc_layer_for_audio_concat)

            fc_layer_for_audio_concat = fc_layer_for_audio_concat.cuda()

        elif args.model_params["intra_modal_fusion"] == "encoder_plus_self_attention":
            transformer_audio_modality_fusion = Intra_modal_transformer_fusion(
                feat_dim=512,
                num_heads=args.model_params["num_heads"],
                hidden_dim=512,
                num_layers=args.model_params["num_layers"],
            )  # todo : Add 712 as the input dimension

            if torch.cuda.device_count() > 1:
                DLLogger.log(
                    f"Using multi-gpus: {torch.cuda.device_count()} GPUs.")
                transformer_audio_modality_fusion = MyDataParallel(
                    transformer_audio_modality_fusion)

            transformer_audio_modality_fusion = transformer_audio_modality_fusion.cuda()
        else:
            raise NotImplementedError(args.model_params["intra_modal_fusion"])



    elif len(args.model_params["l_audio_backbones"]) > 2:
        raise NotImplementedError('New audio backbones? Excepted ResNet18, '
                                  'and wavLM.')

    elif (len(args.model_params["l_audio_backbones"]) == 1) and (
            'wavLM' in args.model_params["l_audio_backbones"]):
        intra_modal_fusion = args.model_params["intra_modal_fusion"]
        assert intra_modal_fusion in ["feat_concat_fc",
                                      "None"], intra_modal_fusion
        if intra_modal_fusion == 'None':
            assert args.goal == 'PRETRAINING', args.goal

        fc_layer_for_audio_concat = FcLayer(768, 512)

        if torch.cuda.device_count() > 1:
            DLLogger.log(f"Using multi-gpus: {torch.cuda.device_count()} GPUs.")
            fc_layer_for_audio_concat = MyDataParallel(
                fc_layer_for_audio_concat)

        fc_layer_for_audio_concat = fc_layer_for_audio_concat.cuda()



    fc_layer_for_video_concat = None
    transformer_visio_modality_fusion = None
    if len(args.model_params['l_vision_backbones']) == 2:
        assert 'R2D1' in args.model_params['l_vision_backbones']
        assert 'I3D' in args.model_params['l_vision_backbones']

        if args.model_params["intra_modal_fusion"] == "feat_concat_fc":

            fc_layer_for_video_concat = FcLayer(512 + 512, 512)

            if torch.cuda.device_count() > 1:
                DLLogger.log(f"Using multi-gpus: {torch.cuda.device_count()} GPUs.")
                fc_layer_for_video_concat = MyDataParallel(
                    fc_layer_for_video_concat)

            fc_layer_for_video_concat = fc_layer_for_video_concat.cuda()

        elif args.model_params[
            "intra_modal_fusion"] == "encoder_plus_self_attention":
            transformer_visio_modality_fusion = Intra_modal_transformer_fusion(
                feat_dim=512,
                num_heads=args.model_params["num_heads"],
                hidden_dim=512,
                num_layers=args.model_params["num_layers"],
            ).cuda()  # TODO : Add 712 as the input dimension

        else:
            raise NotImplementedError(args.model_params["intra_modal_fusion"])

    elif len(args.model_params['l_vision_backbones']) > 2:
        raise NotImplementedError('New vision backbones? Expected R2D1 and '
                                  'I3D.')

    # Freeze or fientune?

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.children():
        p.train(False)

    if (not args.model_params["freeze_vision_R2D1"]) and "R2D1" in \
            args.model_params['l_vision_backbones']:

        for p in model.vision_r2d1.parameters():
            p.requires_grad = True
        model.vision_r2d1.train(True)

        if model.vision_r2d1_fc is not None:
            for p in model.vision_r2d1_fc.parameters():
                p.requires_grad = True
            model.vision_r2d1_fc.train(True)

        DLLogger.log(fmsg("Vision backbone: R2D1 is being finetuned."))

    elif (args.model_params["freeze_vision_R2D1"]) and "R2D1" in \
            args.model_params['l_vision_backbones']:

        if (model.vision_r2d1_fc is not None) and (mode == "Training") and (
            args.model_params['init_w_R2D1'] in [
            'RANDOM', 'KINETICS400', 'AFFWILD2']):
            raise ValueError("Currently `vision_r2d1_fc` is not pretrained."
                             "We cant freeze it while training.")
        # todo: add new init_w with our pretraining.

        DLLogger.log(fmsg("Vision backbone: R2D1 is frozen."))

    if (not args.model_params["freeze_audio_ResNet18"]) and "ResNet18" in \
            args.model_params['l_audio_backbones']:

        for p in model.audio_resnet18.parameters():
            p.requires_grad = True

        model.audio_resnet18.train(True)

        DLLogger.log(fmsg("Audio backbone: ResNet18 is being finetuned."))

    elif (args.model_params["freeze_audio_ResNet18"]) and "ResNet18" in \
            args.model_params['l_audio_backbones']:

        DLLogger.log(fmsg("Audio backbone: ResNet18 is frozen."))

    if (not args.model_params["freeze_vision_I3D"]) and "I3D" in \
            args.model_params['l_vision_backbones']:

        for p in model.vision_i3d.parameters():
            p.requires_grad = True

        model.vision_i3d.train(True)

        DLLogger.log(fmsg("Vision backbone: i3D is being finetuned."))

    elif (args.model_params["freeze_vision_I3D"]) and "I3D" in \
            args.model_params['l_vision_backbones']:

        DLLogger.log(fmsg("Vision backbone: I3D is frozen."))

    ## Fusion model

    # R2D1 features dim reduction:
    _dim = 512
    fusion_model = None
    backbone_pretrainer = None

    if args.goal == 'TRAINING':

        fusion_model = Two_transformers(
            v_dropout=args.model_params["v_dropout"],
            a_dropout=args.model_params["a_dropout"],
            num_heads=args.model_params["num_heads"],
            num_layers=args.model_params["num_layers"],
            joint_modalities=args.model_params['joint_modalities'],
            output_format=args.model_params['output_format'],
            vision_in_ft=_dim
        )

        if torch.cuda.device_count() > 1:
            DLLogger.log(f"Using multi-gpus: {torch.cuda.device_count()} GPUs.")
            fusion_model = MyDataParallel(fusion_model)

        fusion_model = fusion_model.cuda()

    elif args.goal == "PRETRAINING":
        backbone_pretrainer = SingleBackbonePretrainer(
            v_dropout=args.model_params["v_dropout"],
            a_dropout=args.model_params["a_dropout"]
        )

        if torch.cuda.device_count() > 1:
            DLLogger.log(f"Using multi-gpus: {torch.cuda.device_count()} GPUs.")
            backbone_pretrainer = MyDataParallel(backbone_pretrainer)

        backbone_pretrainer = backbone_pretrainer.cuda()


    DLLogger.log(fmsg(f"Mode: {mode}"))

    if mode == "Eval":
        _fd_exp = eval_config["fd_exp"]
        # fusion
        if fusion_model is not None:
            assert args.goal == 'TRAINING', args.goal

            assert os.path.isdir(_fd_exp), _fd_exp
            cam_model_path = join(_fd_exp, "SavedWeights/fusion_w.pt")
            device = next(fusion_model.parameters()).device
            cam_w = load_clean_weights(cam_model_path, map_location=device)
            fusion_model.load_state_dict(cam_w, strict=True)
            DLLogger.log(f"Loaded fusion weights: {cam_model_path}. "
                         f"Goal: {args.goal}")

        # backbones
        model_path = join(_fd_exp, "SavedWeights/all_backbones.pt")

        if len(list(model.parameters())) > 0:
            device = next(model.parameters()).device
        else:
            # assuming the model is on cuda.
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        model_w = load_clean_weights(model_path, map_location=device)
        model.load_state_dict(model_w, strict=True)

        DLLogger.log(f"Loaded all backbones weights: {model_path}.")

        if backbone_pretrainer is not None:
            assert args.goal == 'PRETRAINING', args.goal

            assert os.path.isdir(_fd_exp), _fd_exp
            _path = join(_fd_exp, "SavedWeights/backbone_pretrainer_w.pt")
            device = next(backbone_pretrainer.parameters()).device
            _w = load_clean_weights(_path, map_location=device)
            backbone_pretrainer.load_state_dict(_w, strict=True)
            DLLogger.log(f"Loaded backbone pretrainer weights: {_path}"
                         f"Goal: {args.goal}")


        if fc_layer_for_audio_concat is not None:
            _path = join(_fd_exp, "SavedWeights/fc_layer_for_audio_concat.pt")
            device = next(fc_layer_for_audio_concat.parameters()).device
            _w = load_clean_weights(_path, map_location=device)
            fc_layer_for_audio_concat.load_state_dict(_w, strict=True)

            DLLogger.log(f"Loaded FcLayer audio weights: {_path}")
            assert transformer_audio_modality_fusion is None

        if transformer_audio_modality_fusion is not None:
            _path = join(_fd_exp,
                         "SavedWeights/transformer_audio_modality_fusion.pt")
            device = next(transformer_audio_modality_fusion.parameters()).device
            _w = load_clean_weights(_path, map_location=device)
            transformer_audio_modality_fusion.load_state_dict(_w, strict=True)

            DLLogger.log(f"Loaded Transformer audio modality fusion weights:"
                         f" {_path}")
            assert fc_layer_for_audio_concat is None


        if fc_layer_for_video_concat is not None:
            _path = join(_fd_exp, "SavedWeights/fc_layer_for_video_concat.pt")
            device = next(fc_layer_for_video_concat.parameters()).device
            _w = load_clean_weights(_path, map_location=device)
            fc_layer_for_video_concat.load_state_dict(_w, strict=True)

            DLLogger.log(f"Loaded FcLayer video weights: {_path}")
            assert transformer_visio_modality_fusion is None

        if transformer_visio_modality_fusion is not None:
            _path = join(_fd_exp,
                         "SavedWeights/transformer_visio_modality_fusion.pt")
            device = next(transformer_visio_modality_fusion.parameters()).device
            _w = load_clean_weights(_path, map_location=device)
            transformer_visio_modality_fusion.load_state_dict(_w, strict=True)

            DLLogger.log(f"Loaded FcLayer video weights: {_path}")
            assert fc_layer_for_video_concat is None


        model.eval()
        if fusion_model is not None:
            fusion_model.eval()

        if backbone_pretrainer is not None:
            backbone_pretrainer.eval()

        if fc_layer_for_audio_concat is not None:
            fc_layer_for_audio_concat.eval()
            assert transformer_audio_modality_fusion is None

        if transformer_audio_modality_fusion is not None:
            transformer_audio_modality_fusion.eval()
            assert fc_layer_for_audio_concat is None

        if fc_layer_for_video_concat is not None:
            fc_layer_for_video_concat.eval()
            assert transformer_visio_modality_fusion is None

        if transformer_visio_modality_fusion is not None:
            transformer_visio_modality_fusion.eval()
            assert fc_layer_for_video_concat is None

        _params = list(model.parameters())

        if fusion_model is not None:
            _params = _params + list(fusion_model.parameters())

        if backbone_pretrainer is not None:
            _params = _params + list(backbone_pretrainer.parameters())

        if fc_layer_for_audio_concat is not None:
            _params = _params + list(fc_layer_for_audio_concat.parameters())

        if transformer_audio_modality_fusion is not None:
            _params = _params + list(transformer_audio_modality_fusion.parameters())

        if fc_layer_for_video_concat is not None:
            _params = _params + list(fc_layer_for_video_concat.parameters())

        if transformer_visio_modality_fusion is not None:
            _params = _params + list(transformer_visio_modality_fusion.parameters())

        for param in _params:
            param.requires_grad = False

    if mode == "Training":

        print("Train Data")
        traindataset = ImageList(
            root=args.dataset_rootpath,
            fileList=args.train_params["labelpath"],
            audList=args.dataset_wavspath,
            length=args.train_params["seq_length"],
            flag="train",
            stride=args.train_params["stride"],
            dilation=args.train_params["dilation"],
            subseq_length=args.train_params["subseq_length"],
            realtimestamps_path=args.dataset_realtimestamps,
            use_more_vision_data_augm=args.train_params["use_more_vision_data_augm"],
            use_more_audio_data_augm=args.train_params["use_more_audio_data_augm"],
            take_n_videos=args.train_params['take_n_videos']
        )

        trainloader = torch.utils.data.DataLoader(
            traindataset,
            collate_fn=TrainPadSequence(),
            drop_last=True,  # for DP issue.
            **args.train_params["loader_params"]
        )


        print("Val Data")
        valdataset = ImageList_val(
            root=args.dataset_rootpath,
            fileList=args.val_params["labelpath"],
            audList=args.dataset_wavspath,
            length=args.val_params["seq_length"],
            flag="val",
            stride=args.val_params["stride"],
            dilation=args.val_params["dilation"],
            subseq_length=args.val_params["subseq_length"],
            realtimestamps_path=args.dataset_realtimestamps,
            take_n_videos=args.val_params['take_n_videos']
        )

        valloader = torch.utils.data.DataLoader(
            valdataset, collate_fn=ValPadSequence(), **args.val_params["loader_params"]
        )

        print("Number of Train samples:" + str(len(traindataset)))
        print("Number of Val samples:" + str(len(valdataset)))

    elif mode == "Eval":

        _fd_exp = eval_config["fd_exp"]
        _eval_set = eval_config["eval_set"]
        store_results_pkl = join(_fd_exp, f"{_eval_set}-reevaluation.pkl")

        if eval_config["eval_set"] == "test":

            DLLogger.log("Testing...")
            testdataset = ImageList_test(
                root=args.dataset_rootpath,
                fileList=args.test_params["labelpath"],
                audList=args.dataset_wavspath,
                length=args.test_params["seq_length"],
                flag="Test",
                stride=args.test_params["stride"],
                dilation=args.test_params["dilation"],
                subseq_length=args.test_params["subseq_length"],
                realtimestamps_path=args.dataset_realtimestamps
            )

            testloader = torch.utils.data.DataLoader(
                testdataset,
                collate_fn=TestPadSequence(),
                **args.test_params["loader_params"],
            )
            print("Number of Test samples:" + str(len(testdataset)))
            _dir_out = join(_fd_exp, f"eval-{eval_config['eval_set']}")
            os.makedirs(_dir_out, exist_ok=True)
            test_tic = dt.datetime.now()
            Test(
                testloader,
                model,
                fusion_model,
                args,
                store_results_pkl=store_results_pkl,
                dir_out=_dir_out,
                fc_layer_for_audio_concat=fc_layer_for_audio_concat,
                transformer_audio_modality_fusion
                =transformer_audio_modality_fusion,
                fc_layer_for_video_concat=fc_layer_for_video_concat,
                transformer_visio_modality_fusion
                =transformer_visio_modality_fusion,
                backbone_pretrainer=backbone_pretrainer
            )
            
            test_toc = dt.datetime.now()
            DLLogger.log(f"Eval time for {_eval_set} set: {test_toc - test_tic}")
            DLLogger.log(fmsg("bye."))
            DLLogger.flush()
            sys.exit()

        elif eval_config["eval_set"] == "valid":

            take_n_videos = args.val_params['take_n_videos']
            if take_n_videos != -1:
                take_n_videos = -1
                DLLogger.log(f"We have switched `take_n_videos` from "
                             f"{args.val_params['take_n_videos']} to -1 for "
                             f"validset for mode = {mode}.")

            valdataset = ImageList_val(
                root=args.dataset_rootpath,
                fileList=args.val_params["labelpath"],
                audList=args.dataset_wavspath,
                length=args.val_params["seq_length"],
                flag="val",
                stride=args.val_params["stride"],
                dilation=args.val_params["dilation"],
                subseq_length=args.val_params["subseq_length"],
                realtimestamps_path=args.dataset_realtimestamps,
                take_n_videos=take_n_videos
            )
            valloader = torch.utils.data.DataLoader(
                valdataset,
                collate_fn=ValPadSequence(),
                **args.val_params["loader_params"],
            )

            set_seed(default_seed)
            val_tic = dt.datetime.now()
            # todo: fix the arguments.
            Valid_vacc, Valid_aacc = validate(
                valloader,
                model,
                fusion_model,
                args,
                store_results_pkl=store_results_pkl,
                fc_layer_for_audio_concat=fc_layer_for_audio_concat,
                transformer_audio_modality_fusion
                =transformer_audio_modality_fusion,
                fc_layer_for_video_concat=fc_layer_for_video_concat,
                transformer_visio_modality_fusion
                =transformer_visio_modality_fusion,
                backbone_pretrainer=backbone_pretrainer
            )
            DLLogger.log(fmsg("Final results:"))
            DLLogger.log(f"{_eval_set}: {_AROUSAL}: [BEST: {Valid_aacc:.4f}]")
            DLLogger.log(f"{_eval_set}: {_VALENCE}: [BEST: {Valid_vacc:.4f}]")

            val_toc = dt.datetime.now()
            DLLogger.log(f"Eval time for {_eval_set} set: {val_toc - val_tic}")

            DLLogger.log(fmsg("bye."))
            DLLogger.flush()
            sys.exit()

        elif eval_config["eval_set"] == "train":
            raise NotImplementedError(eval_config["eval_set"])

    else:
        raise NotImplementedError(mode)

    criterion = CCCLoss(digitize_num=1).cuda()

    l_params: list = []
    if fusion_model is not None:
        l_params = list(fusion_model.parameters())

    if backbone_pretrainer is not None:
        l_params = l_params + list(backbone_pretrainer.parameters())

    if (not args.model_params["freeze_vision_R2D1"]) and "R2D1" in \
            args.model_params['l_vision_backbones']:
        l_params = l_params + list(model.vision_r2d1.parameters())
        if model.vision_r2d1_fc is not None:
            l_params = l_params + list(model.vision_r2d1_fc.parameters())

    if (not args.model_params["freeze_vision_I3D"]) and "I3D" in \
            args.model_params['l_vision_backbones']:
        l_params = l_params + list(model.vision_i3d.parameters())

    if (not args.model_params["freeze_audio_ResNet18"]) and "ResNet18" in \
            args.model_params['l_audio_backbones']:
        l_params = l_params + list(model.audio_resnet18.parameters())


    if fc_layer_for_audio_concat is not None:
        l_params = l_params + list(fc_layer_for_audio_concat.parameters())

    if transformer_audio_modality_fusion is not None:
        l_params = l_params + list(transformer_audio_modality_fusion.parameters())

    if fc_layer_for_video_concat is not None:
        l_params = l_params + list(fc_layer_for_video_concat.parameters())

    if transformer_visio_modality_fusion is not None:
        l_params = l_params + list(transformer_visio_modality_fusion.parameters())

    assert l_params != []

    optimizer, lrate_scheduler = get_optimizer_for_params(
        args.model_params, l_params, "opt"
    )

    DLLogger.log(fmsg("Start training"))

    best_Val_acc = 0  
    best_Val_acc_epoch = 0
    start_epoch = args.model_params["start_epoch"]  
    total_epoch = args.model_params["max_epochs"]  

    tracker = {
        subset: {
            _VALENCE: [],
            _AROUSAL: [],
            _AVG: [],
            "best_idx": 0,
            "best_val": 0.0,
            "best_ar": 0.0,
            "best_avg": 0.0,
        }
        for subset in [_TRAIN, _VALID]
    }

    best_idx = 0
    best_fusion_model = None
    if fusion_model is not None:
        best_fusion_model = copy.deepcopy(fusion_model)

    best_backbone_pretrainer = None
    if backbone_pretrainer is not None:
        best_backbone_pretrainer = copy.deepcopy(backbone_pretrainer)

    # because of `weight_norm`, we cant do deepcopy.
    best_model = deepcopy_state_dict(model)

    best_fc_layer_for_audio_concat = None
    if fc_layer_for_audio_concat is not None:
        best_fc_layer_for_audio_concat = copy.deepcopy(
            fc_layer_for_audio_concat
        )
        assert transformer_audio_modality_fusion is None

    best_transformer_audio_modality_fusion = None
    if transformer_audio_modality_fusion is not None:
        best_transformer_audio_modality_fusion = copy.deepcopy(
            transformer_audio_modality_fusion
        )
        assert fc_layer_for_audio_concat is None


    best_fc_layer_for_video_concat = None
    if fc_layer_for_video_concat is not None:
        best_fc_layer_for_video_concat = copy.deepcopy(
            fc_layer_for_video_concat
        )

    best_transformer_visio_modality_fusion = None
    if transformer_visio_modality_fusion is not None:
        best_transformer_visio_modality_fusion = copy.deepcopy(
            transformer_visio_modality_fusion
        )
        assert fc_layer_for_video_concat is None

    max_seed = (2**32) - 1

    for epoch in range(start_epoch, total_epoch):
        set_seed(min(epoch + default_seed, max_seed))

        epoch_tic = dt.datetime.now()
        DLLogger.log(f"Epoch : {epoch} / {total_epoch - 1}")

        train_tic = dt.datetime.now()
        Training_vacc, Training_aacc = train(
            trainloader,
            model,
            criterion,
            optimizer,
            lrate_scheduler,
            fusion_model,
            args,
            fc_layer_for_audio_concat=fc_layer_for_audio_concat,
            transformer_audio_modality_fusion=transformer_audio_modality_fusion,
            fc_layer_for_video_concat=fc_layer_for_video_concat,
            transformer_visio_modality_fusion=transformer_visio_modality_fusion,
            backbone_pretrainer=backbone_pretrainer
        )
        train_toc = dt.datetime.now()
        DLLogger.log(f"Train time epoch {epoch}: {train_toc - train_tic}")

        set_seed(default_seed)
        val_tic = dt.datetime.now()
        Valid_vacc, Valid_aacc = validate(
            valloader,
            model,
            fusion_model,
            args,
            fc_layer_for_audio_concat=fc_layer_for_audio_concat,
            transformer_audio_modality_fusion=transformer_audio_modality_fusion,
            fc_layer_for_video_concat=fc_layer_for_video_concat,
            transformer_visio_modality_fusion=transformer_visio_modality_fusion,
            backbone_pretrainer=backbone_pretrainer
        )
        val_toc = dt.datetime.now()
        DLLogger.log(f"Validation time: {val_toc - val_tic}")

        # Train tracker 
        tracker[_TRAIN][_VALENCE].append(Training_vacc.item())
        tracker[_TRAIN][_AROUSAL].append(Training_aacc.item())
        tracker[_TRAIN][_AVG].append(((Training_vacc + Training_vacc) / 2.0).item())
        
        # Validation tracker
        tracker[_VALID][_VALENCE].append(Valid_vacc.item())
        tracker[_VALID][_AROUSAL].append(Valid_aacc.item())
        tracker[_VALID][_AVG].append(((Valid_vacc + Valid_aacc) / 2.0).item())

        if tracker[_VALID][_AVG][-1] >= best_Val_acc:
            if best_fusion_model is not None:
                best_fusion_model = copy.deepcopy(fusion_model)

            if best_backbone_pretrainer is not None:
                best_backbone_pretrainer = copy.deepcopy(backbone_pretrainer)

            if best_model is not None:
                model.flush()
                best_model = deepcopy_state_dict(model)

            if best_fc_layer_for_audio_concat is not None:
                best_fc_layer_for_audio_concat = copy.deepcopy(
                    fc_layer_for_audio_concat
                )

            if transformer_audio_modality_fusion is not None:
                transformer_audio_modality_fusion = copy.deepcopy(
                    transformer_audio_modality_fusion
                )

            if best_fc_layer_for_video_concat is not None:
                best_fc_layer_for_video_concat = copy.deepcopy(
                    fc_layer_for_video_concat
                )

            if best_transformer_visio_modality_fusion is not None:
                best_transformer_visio_modality_fusion = copy.deepcopy(
                    transformer_visio_modality_fusion
                )

            best_Val_acc = tracker[_VALID][_AVG][-1]
            best_idx = len(tracker[_VALID][_AVG]) - 1

            best_Val_acc_epoch = epoch

            if args.dump_best_model_every_time:
                # Save on disk. ------------------------------------------------
                dump_models_into_disk(best_model,
                                      best_fusion_model,
                                      best_backbone_pretrainer,
                                      best_fc_layer_for_audio_concat,
                                      best_transformer_audio_modality_fusion,
                                      best_fc_layer_for_video_concat,
                                      best_transformer_visio_modality_fusion,
                                      epoch,
                                      path)
                # ------------------------- End save models on disk ------------

        DLLogger.log(
            f"Valid {_AROUSAL} @EPOCH {epoch}: "
            f"{tracker[_VALID][_AROUSAL][-1]:.4f} | "
            f"[BEST: {tracker[_VALID][_AROUSAL][best_idx]:.4f} "
            f"@EPOCH: {best_Val_acc_epoch}]"
        )
        DLLogger.log(
            f"Valid {_VALENCE} @EPOCH {epoch}: "
            f"{tracker[_VALID][_VALENCE][-1]:.4f} | "
            f"[BEST: {tracker[_VALID][_VALENCE][best_idx]:.4f} "
            f"@EPOCH: {best_Val_acc_epoch}]"
        )

        epoch_toc = dt.datetime.now()
        DLLogger.log(f"Full epoch {epoch} took: {epoch_toc - epoch_tic}")
        DLLogger.flush()

    # tb.close()
    DLLogger.log(fmsg("Final results"))
    DLLogger.log(
        f"BEST Valid {_AROUSAL}: "
        f"[BEST: {tracker[_VALID][_AROUSAL][best_idx]:.4f} "
        f"@EPOCH: {best_Val_acc_epoch}]"
    )
    DLLogger.log(
        f"BEST Valid {_VALENCE}: "
        f"[BEST: {tracker[_VALID][_VALENCE][best_idx]:.4f} "
        f"@EPOCH: {best_Val_acc_epoch}]"
    )

    # Save on disk. ------------------------------------------------------------
    dump_models_into_disk(best_model,
                          best_fusion_model,
                          best_backbone_pretrainer,
                          best_fc_layer_for_audio_concat,
                          best_transformer_audio_modality_fusion,
                          best_fc_layer_for_video_concat,
                          best_transformer_visio_modality_fusion,
                          best_Val_acc_epoch,
                          path)
    # ------------------------- End save models on disk ------------------------

    for subset in tracker:
        tracker[subset]["best_idx"] = best_idx
        tracker[subset]["best_val"] = tracker[subset][_VALENCE][best_idx]
        tracker[subset]["best_ar"] = tracker[subset][_AROUSAL][best_idx]
        tracker[subset]["best_avg"] = tracker[subset][_AVG][best_idx]

    with open(join(args.outd, "perfs.yml"), "w") as fx:
        yaml.dump(tracker, fx)

    plot_tracker(tracker, fdout=args.outd)

    args.tend = dt.datetime.now()
    DLLogger.log(fmsg("End time: {}".format(args.tend)))
    DLLogger.log(fmsg("Total time: {}".format(args.tend - args.t0)))

    with open(join(args.outd, "final_config.yml"), "w") as f:
        _args = copy.deepcopy(args)
        _args.model_params["l_vision_backbones"] = "+".join(
            _args.model_params["l_vision_backbones"]
        )
        if _args.model_params["l_vision_backbones"] == '':
            _args.model_params["l_vision_backbones"] = 'None'

        _args.model_params["l_audio_backbones"] = "+".join(
            _args.model_params["l_audio_backbones"]
        )

        if _args.model_params["l_audio_backbones"] == '':
            _args.model_params["l_audio_backbones"] = 'None'

        yaml.dump(vars(_args), f)

        with open(join(args.outd, "SavedWeights", "config.yml"), "w") as fyaml:
            yaml.dump(vars(_args), fyaml)

    with open(join(args.outd, "passed.txt"), "w") as fout:
        fout.write("Passed.")

    DLLogger.log(fmsg("bye."))


if __name__ == "__main__":
    main()
