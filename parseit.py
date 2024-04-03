# Sel-contained-as-possible module handles parsing the input using argparse.
# handles seed, and initializes some modules for reproducibility.
import copy
import os
from os.path import dirname, abspath, join, basename, expanduser, normpath
import sys
import argparse
from copy import deepcopy
import warnings
import subprocess
import fnmatch
import glob
import shutil
import datetime as dt
from typing import Tuple
import socket
import getpass

import yaml
import json
import numpy as np
import torch

root_dir = dirname((abspath(__file__)))
sys.path.append(root_dir)

import reproducibility

from dllogger import ArbJSONStreamBackend
from dllogger import Verbosity
from dllogger import ArbStdOutBackend
from dllogger import ArbTextStreamBackend
import dllogger as DLLogger
from tools import fmsg


def mkdir(fd):
    if not os.path.isdir(fd):
        os.makedirs(fd, exist_ok=True)


def find_files_pattern(fd_in_, pattern_):
    assert os.path.exists(fd_in_), "Folder {} does not exist.".format(fd_in_)
    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def null_str(v):
    if v in [None, '', 'None']:
        return 'None'

    if isinstance(v, str):
        return v

    raise NotImplementedError(f"{v}, type: {type(v)}")


class Dict2Obj(object):
    def __init__(self, dictionary):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s" % attrs


def get_args(args: dict) -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument("--cudaid", type=str, default=None,
                        help="cuda id: '0,1,2,3'")
    parser.add_argument("--SEED", type=int, default=None, help="Seed.")
    parser.add_argument("--goal", type=str, default=None,
                        help="The goal: 'TRAINING': standard train model for "
                             "fusion. "
                             "Requires 2 modalities == 2 backbones at least. "
                             "'PRETRAINING': used to pretrain SINGLE backbone "
                             "== SINGLE modality."
                        )
    parser.add_argument("--Mode", type=str, default=None,
                        help="Mode: 'Training', 'Eval'.")
    parser.add_argument("--verbose", type=str2bool, default=None,
                        help="Verbosity (bool).")

    # model_params -> lr, max_epochs
    parser.add_argument("--output_format", type=str,
                        default=None,
                        help="The type of the final layer in the model: 'FC' "
                             "using a fully connected layer. 'SELF_ATTEN' "
                             ": use an transformer.")
    parser.add_argument("--intra_modal_fusion", type=str,
                        default=None,
                        help="How to fuse features from different backbones "
                             "from the same modality. options: "
                             "`feat_concat_fc`: use a fully connected layer "
                             "with concatenated features."
                             "`encoder_plus_self_attention`: use a "
                             "transformer with stacked features. These "
                             "options can be used only when "
                             "`goal=TRAINING`. If `goal=PRETRAINING`, "
                             "this must be set to `None`.")
    parser.add_argument("--joint_modalities", type=str,
                        default=None,
                        help="How to build the joint modality: `NONE`, "
                             "`FC`, `TRANFORMER`.")
    parser.add_argument("--l_vision_backbones", type=str, default=None,
                        help="List of vision modality backbones. Accepted: "
                             "'R2D1', 'I3D'. To combine many, separate them "
                             "with '+': eg. 'R2D1+I3D'. To indicate None, "
                             "use the string 'None' for now vision backbone.")
    parser.add_argument("--l_audio_backbones", type=str, default=None,
                        help="List of audio modality backbones. Accepted: "
                             "'ResNet18', 'wavLM'. To combine many, separate "
                             "them with '+': eg. 'ResNet18+wavLM'. To "
                             "indicate None, use the string 'None'.")
    parser.add_argument("--init_w_R2D1", type=str, default=None,
                        help="Initial weights of R2D1 model: 'RANDOM', "
                             "'KINETICS400', 'AFFWILD2', 'OUR_AFFWILD2'. "
                             "Vision backbone.")
    parser.add_argument("--init_w_I3D", type=str, default=None,
                        help="Initial weights of I3D model: 'RANDOM', "
                             "'KINETICS400', 'OUR_AFFWILD2'. Vision backbone.")
    parser.add_argument("--init_w_ResNet18", type=str, default=None,
                        help="Initial weights of ResNet18 model: 'RANDOM', "
                             "'IMAGENET', 'AFFWILD2', 'OUR_AFFWILD2'. "
                             "Audio backbone.")
    parser.add_argument("--R2D1_ft_dim_reduce", type=str, default=None,
                        help="How to reduce 2d features of R2D1: `MAX`, "
                             "`AVG`, `FLATTEN`.")
    parser.add_argument("--split", type=str, default=None,
                        help="Data split: 'DEFAULT', 'ROUND1' to 'ROUND5'.")
    parser.add_argument("--dump_best_model_every_time", type=str2bool,
                        default=None,
                        help="If True, the best found model at each epoch is "
                             "stored on disk. Otherwise, storing on disk is "
                             "performed only when training is done.")
    parser.add_argument("--freeze_vision_R2D1", type=str2bool, default=None,
                        help="Whether to freeze of not vision R2D1 backbone.")
    parser.add_argument("--freeze_vision_I3D", type=str2bool, default=None,
                        help="Whether to freeze of not vision I3D backbone.")
    parser.add_argument("--freeze_audio_ResNet18", type=str2bool, default=None,
                        help="Whether to freeze of not ResNet18 audio "
                             "backbone.")
    parser.add_argument("--num_layers", type=int, default=None,
                        help="Number of layers in transformers in the fusion "
                             "module.")
    parser.add_argument("--v_dropout", type=float, default=None,
                        help="Dropout for valence module in Two_transformers.")
    parser.add_argument("--a_dropout", type=float, default=None,
                        help="Dropout for arousal module in Two_transformers.")
    parser.add_argument("--num_heads", type=int, default=None,
                        help="Number of heads in transformers in the fusion "
                             "module.")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Max epochs.")
    parser.add_argument("--exp_id", type=str, default=None, help="Exp id.")
    parser.add_argument("--start_epoch", type=int, default=None,
                        help="Start epoch - in case of restarting training.")

    parser.add_argument("--opt__name_optimizer", type=str, default=None,
                        help="Name of the optimizer 'sgd', 'adam'.")
    parser.add_argument("--opt__lr", type=float, default=None,
                        help="Learning rate (optimizer)")
    parser.add_argument("--opt__momentum", type=float, default=None,
                        help="Momentum (optimizer)")
    parser.add_argument("--opt__dampening", type=float, default=None,
                        help="Dampening for Momentum (optimizer)")
    parser.add_argument("--opt__nesterov", type=str2bool, default=None,
                        help="Nesterov or not for Momentum (optimizer)")
    parser.add_argument("--opt__weight_decay", type=float, default=None,
                        help="Weight decay (optimizer)")
    parser.add_argument("--opt__beta1", type=float, default=None,
                        help="Beta1 for adam (optimizer)")
    parser.add_argument("--opt__beta2", type=float, default=None,
                        help="Beta2 for adam (optimizer)")
    parser.add_argument("--opt__eps_adam", type=float, default=None,
                        help="eps for adam (optimizer)")
    parser.add_argument("--opt__amsgrad", type=str2bool, default=None,
                        help="amsgrad for adam (optimizer)")
    parser.add_argument("--opt__lr_scheduler", type=str2bool, default=None,
                        help="Whether to use or not a lr scheduler")
    parser.add_argument("--opt__name_lr_scheduler", type=str, default=None,
                        help="Name of the lr scheduler")
    parser.add_argument("--opt__gamma", type=float, default=None,
                        help="Gamma of the lr scheduler. (mystep)")
    parser.add_argument("--opt__last_epoch", type=int, default=None,
                        help="Index last epoch to stop adjust LR(mystep)")
    parser.add_argument("--opt__min_lr", type=float, default=None,
                        help="Minimum allowed value for lr.")
    parser.add_argument("--opt__t_max", type=float, default=None,
                        help="T_max, maximum epochs to restart. (cosine)")
    parser.add_argument("--opt__mode", type=float, default=None,
                        help="T_max, maximum epochs to restart. "
                             "(reduceonplateau)")
    parser.add_argument("--opt__factor", type=float, default=None,
                        help="Factor, factor by which learning rate is "
                             "reduced. (reduceonplateau)")
    parser.add_argument("--opt__patience", type=int, default=None,
                        help="Patience, number of epoch with no improvement. ("
                             "reduceonplateau)")
    parser.add_argument("--opt__step_size", type=int, default=None,
                        help="Step size for lr scheduler.")


    # train_params: if you want to change a variable inside "train_params",
    # the variable 'X' should be named here: "train_params__X".
    parser.add_argument("--train_params__take_n_videos", type=int, default=None,
                        help="Train_params: take_n_videos. how many train "
                             "videos to consider. -1 to use all.")
    parser.add_argument("--train_params__seq_length", type=int, default=None,
                        help="Train_params: seq_length.")
    parser.add_argument("--train_params__subseq_length", type=int, default=None,
                        help="Train_params: subseq_length.")
    parser.add_argument("--train_params__stride", type=int, default=None,
                        help="Train_params: stride.")
    parser.add_argument("--train_params__dilation", type=int, default=None,
                        help="Train_params: dilation.")
    parser.add_argument("--train_params__use_more_vision_data_augm",
                        type=str2bool, default=None,
                        help="Train_params: use_more_vision_data_augm.")
    parser.add_argument("--train_params__use_more_audio_data_augm",
                        type=str2bool, default=None,
                        help="Train_params: use_more_audio_data_augm.")
    parser.add_argument("--train_params__batch_size", type=int, default=None,
                        help="Train_params: batch_size.")
    parser.add_argument("--train_params__num_workers", type=int, default=None,
                        help="Train_params: num_workers.")
    parser.add_argument("--train_params__pin_memory", type=str2bool,
                        default=None,
                        help="Train_params: pin_memory.")
    parser.add_argument("--train_params__shuffle", type=str2bool,
                        default=None,
                        help="Train_params: shuffle.")

    # val_params:
    parser.add_argument("--val_params__take_n_videos", type=int, default=None,
                        help="val_params: take_n_videos. how may videos to "
                             "consider for validation set. -1 to use all.")
    parser.add_argument("--val_params__seq_length", type=int, default=None,
                        help="val_params: seq_length.")
    parser.add_argument("--val_params__subseq_length", type=int, default=None,
                        help="val_params: subseq_length.")
    parser.add_argument("--val_params__stride", type=int, default=None,
                        help="val_params: stride.")
    parser.add_argument("--val_params__dilation", type=int, default=None,
                        help="val_params: dilation.")
    parser.add_argument("--val_params__batch_size", type=int, default=None,
                        help="val_params: batch_size.")
    parser.add_argument("--val_params__num_workers", type=int, default=None,
                        help="val_params_: num_workers.")
    parser.add_argument("--val_params__pin_memory", type=str2bool,
                        default=None,
                        help="val_params_: pin_memory.")

    # test_params
    parser.add_argument("--test_params__seq_length", type=int, default=None,
                        help="test_params: seq_length.")
    parser.add_argument("--test_params__subseq_length", type=int, default=None,
                        help="test_params: subseq_length.")
    parser.add_argument("--test_params__stride", type=int, default=None,
                        help="test_params: stride.")
    parser.add_argument("--test_params__dilation", type=int, default=None,
                        help="test_params: dilation.")
    parser.add_argument("--test_params__batch_size", type=int, default=None,
                        help="test_params: batch_size.")
    parser.add_argument("--test_params__num_workers", type=int, default=None,
                        help="test_params: num_workers.")
    parser.add_argument("--test_params__pin_memory", type=str2bool,
                        default=None,
                        help="test_params: pin_memory.")


    input_parser = parser.parse_args()

    attributes = input_parser.__dict__.keys()

    for k in attributes:
        val_k = getattr(input_parser, k)
        if k in args.keys():
            if val_k is not None:
                args[k] = val_k

        elif k in args['model_params'].keys():  # try model_params
            if val_k is not None:
                args['model_params'][k] = val_k

        elif k.startswith('test_params'):  # test_params
            _k = k.split('__')[1]
            if _k in args['test_params'].keys():
                if val_k is not None:
                    args['test_params'][_k] = val_k

            elif _k in args['test_params']['loader_params'].keys():
                if val_k is not None:
                    args['test_params']['loader_params'][_k] = val_k

            else:
                raise NotImplementedError(f'Unknown key {k}')

        elif k.startswith('val_params'):  # val_params
            _k = k.split('__')[1]
            if _k in args['val_params'].keys():
                if val_k is not None:
                    args['val_params'][_k] = val_k

            elif _k in args['val_params']['loader_params'].keys():
                if val_k is not None:
                    args['val_params']['loader_params'][_k] = val_k

            else:
                raise NotImplementedError(f'Unknown key {k}')

        elif k.startswith('train_params'):  # train_params
            _k = k.split('__')[1]
            if _k in args['train_params'].keys():
                if val_k is not None:
                    args['train_params'][_k] = val_k

            elif _k in args['train_params']['loader_params'].keys():
                if val_k is not None:
                    args['train_params']['loader_params'][_k] = val_k

            else:
                raise NotImplementedError(f'Unknown key {k}')

        else:
            raise ValueError(f"Key {k} was not found in args. ... [NOT OK]")

    os.environ['MYSEED'] = str(args["SEED"])
    max_seed = (2 ** 32) - 1
    msg = f"seed must be: 0 <= {int(args['SEED'])} <= {max_seed}"
    assert 0 <= int(args['SEED']) <= max_seed, msg

    args['outd'] = outfd(Dict2Obj(copy.deepcopy(args)))

    cmdr = os.path.isfile(join(args['outd'], 'passed.txt'))
    if cmdr:
        warnings.warn('EXP {} has already been done. EXITING.'.format(
            args['outd']))
        sys.exit(0)

    torch.cuda.set_device(0)
    args['cudaid'] = 0

    # --
    l_vision_backbones = args['model_params']['l_vision_backbones'].split('+')
    assert len(l_vision_backbones) == len(set(l_vision_backbones))
    for bk in l_vision_backbones:
        assert bk in ['R2D1', 'I3D', 'None'], bk

    if 'None' in l_vision_backbones:
        assert len(l_vision_backbones) == 1, len(l_vision_backbones)

    if l_vision_backbones == ['None']:
        l_vision_backbones = []


    args['model_params']['l_vision_backbones']: list = l_vision_backbones

    l_audio_backbones = args['model_params']['l_audio_backbones'].split('+')
    assert len(l_audio_backbones) == len(set(l_audio_backbones))
    for bk in l_audio_backbones:
        assert bk in ['ResNet18', 'wavLM', 'None'], bk

    if 'None' in l_audio_backbones:
        assert len(l_audio_backbones) == 1, len(l_audio_backbones)

    if l_audio_backbones == ['None']:
        l_audio_backbones = []

    args['model_params']['l_audio_backbones']: list = l_audio_backbones

    n_backbones = len(l_vision_backbones) + len(l_audio_backbones)

    assert args['goal'] in ['PRETRAINING', 'TRAINING'], args['goal']

    intra_modal_fusion = args['model_params']['intra_modal_fusion']
    assert intra_modal_fusion in ['None', 'feat_concat_fc',
                                  'encoder_plus_self_attention' ], intra_modal_fusion

    if intra_modal_fusion in ['encoder_plus_self_attention', 'feat_concat_fc']:
        assert any([len(l_audio_backbones) == 2, len(l_vision_backbones) == 2])
        if len(l_audio_backbones) == 2:
            assert 'ResNet18' in l_audio_backbones
            assert 'wavLM' in l_audio_backbones

        if len(l_vision_backbones) == 2:
            assert 'R2D1' in l_vision_backbones
            assert 'I3D' in l_vision_backbones

    joint_modalities = args['model_params']['joint_modalities']

    if args['goal'] == 'PRETRAINING':
        assert n_backbones == 1, n_backbones  # pretrain one backbone at once.
        assert intra_modal_fusion == 'None', intra_modal_fusion
        assert joint_modalities == 'NONE', joint_modalities


    elif args['goal'] == 'TRAINING':
        assert len(l_vision_backbones) > 0, len(l_vision_backbones)
        assert len(l_audio_backbones) > 0, len(l_audio_backbones)
        assert n_backbones > 1, n_backbones

        assert joint_modalities in ['NONE', 'TRANSFORMER', 'FC'], \
            joint_modalities

    else:
        raise NotImplementedError(args['goal'])


    output_format = args['model_params']['output_format']
    assert output_format in ['FC', 'SELF_ATTEN'], output_format

    assert args['split'] in ['DEFAULT',
                             'ROUND1',
                             'ROUND2',
                             'ROUND3',
                             'ROUND4',
                             'ROUND5'], args['split']

    args = auto_set_tr_vl_tst_paths(args)


    freeze_vision_R2D1 = args['model_params']['freeze_vision_R2D1']
    assert isinstance(freeze_vision_R2D1, bool), type(freeze_vision_R2D1)
    if not freeze_vision_R2D1:
        assert 'R2D1' in l_vision_backbones

    freeze_vision_I3D = args['model_params']['freeze_vision_I3D']
    assert isinstance(freeze_vision_I3D, bool), type(freeze_vision_I3D)
    if not freeze_vision_I3D:
        assert 'I3D' in l_vision_backbones

    freeze_audio_ResNet18 = args['model_params']['freeze_audio_ResNet18']
    assert isinstance(freeze_audio_ResNet18, bool), type(freeze_audio_ResNet18)
    if not freeze_audio_ResNet18:
        assert 'ResNet18' in l_audio_backbones

    R2D1_ft_dim_reduce = args['model_params']['R2D1_ft_dim_reduce']
    assert R2D1_ft_dim_reduce in ['FLATTEN', 'MAX', 'AVG'], R2D1_ft_dim_reduce


    init_w_R2D1 = args['model_params']['init_w_R2D1']
    assert init_w_R2D1 in ['RANDOM', 'KINETICS400', 'AFFWILD2',
                           'OUR_AFFWILD2'], init_w_R2D1

    init_w_I3D = args['model_params']['init_w_I3D']
    assert init_w_I3D in ['RANDOM', 'KINETICS400', 'AFFWILD2',
                          'OUR_AFFWILD2'], init_w_I3D

    init_w_ResNet18 = args['model_params']['init_w_ResNet18']
    assert init_w_ResNet18 in ['RANDOM', 'IMAGENET', 'AFFWILD2',
                               'OUR_AFFWILD2'], init_w_ResNet18

    reproducibility.set_to_deterministic(seed=int(args["SEED"]), verbose=True)

    return args




def auto_set_tr_vl_tst_paths(args: dict) -> dict:
    split = args['split']
    assert split in ['DEFAULT',
                     'ROUND1',
                     'ROUND2',
                     'ROUND3',
                     'ROUND4',
                     'ROUND5'], split

    _DEFAULT = '/projets2/AS84330/Datasets/Affwild/VA_annotations/'

    _IDS = {
        'ROUND1': 'fold1',
        'ROUND2': 'fold2',
        'ROUND3': 'fold3',
        'ROUND4': 'fold4',
        'ROUND5': 'fold5'
    }
    _FOLD_BASE = '/projets2/AS84330/Datasets/Affwild/VA_annotations_5folds/'

    if split == 'DEFAULT':
        fold_p = _DEFAULT

    else:
        fold_p = join(_FOLD_BASE, _IDS[split])


    args['train_params']['labelpath'] = join(fold_p, 'Train_Set')
    args['val_params']['labelpath'] = join(fold_p, 'Val_Set')
    args['test_params']['labelpath'] = join(fold_p, 'Test_Set')

    for v in ['train_params', 'val_params', 'test_params']:
        assert os.path.isdir(args[v]['labelpath']), args[v]['labelpath']

    return args


def outfd(args):
    tag = [('id', args.exp_id)]

    tag = [(el[0], str(el[1])) for el in tag]
    tag = '-'.join(['_'.join(el) for el in tag])

    parent_lv = "exps"
    subpath = join(parent_lv, tag)
    outd = join(root_dir, subpath)

    outd = expanduser(outd)
    os.makedirs(outd, exist_ok=True)

    return outd


def wrap_sys_argv_cmd(cmd: str, pre):
    splits = cmd.split(' ')
    el = splits[1:]
    pairs = ['{} {}'.format(i, j) for i, j in zip(el[::2], el[1::2])]
    pro = splits[0]
    sep = ' \\\n' + (len(pre) + len(pro) + 2) * ' '
    out = sep.join(pairs)
    return "{} {} {}".format(pre, pro, out)


def get_tag_device(args: dict) -> str:
    tag = ''

    if torch.cuda.is_available():
        txt = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
        try:
            cudaids = args['cudaid'].split(',')
            tag = 'CUDA devices: \n'
            for cid in cudaids:
                tag += 'ID: {} - {} \n'.format(cid, txt[int(cid)])
        except IndexError:
            tag = 'CUDA devices: lost.'

    return tag


def parse_input(config_file: str) -> Tuple[dict, str, dict]:

    # Mandatory
    parser = argparse.ArgumentParser()

    parser.add_argument("--Mode", type=str,
                        help="Mode: Training, Eval")
    input_args, _ = parser.parse_known_args()

    mode = input_args.Mode
    assert mode in ['Training', 'Eval'], mode

    eval_config = {
        'eval_set': '',
        'fd_exp': ''
    }

    if mode == 'Training':

        with open(config_file, 'r') as jsonfile:
            args: dict = json.load(jsonfile)

        args['t0'] = dt.datetime.now()

        args: dict = get_args(args)

        log_backends = [
            ArbJSONStreamBackend(Verbosity.VERBOSE,
                                 join(args['outd'], "log.json")),
            ArbTextStreamBackend(Verbosity.VERBOSE,
                                 join(args['outd'], "log.txt")),
        ]

        if args['verbose']:
            log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))

        DLLogger.init_arb(backends=log_backends, master_pid=os.getpid())

        DLLogger.log(fmsg("Start time: {}".format(args['t0'])))

        DLLogger.log(fmsg(f"Fold: {args['split']}"))

        outd = args['outd']

        with open(join(outd, "config.yml"), 'w') as fyaml:
            _args = copy.deepcopy(args)

            _args['model_params']['l_vision_backbones'] = "+".join(
                _args['model_params']['l_vision_backbones'])

            _args['model_params']['l_audio_backbones'] = "+".join(
                _args['model_params']['l_audio_backbones'])

            yaml.dump(args, fyaml)

        str_cmd = wrap_sys_argv_cmd(" ".join(sys.argv), "time python")
        with open(join(outd, "cmd.sh"), 'w') as frun:
            frun.write("#!/usr/bin/env bash \n")
            frun.write(str_cmd)

        # Announce init w:
        if 'R2D1' in args['model_params']['l_vision_backbones']:
            DLLogger.log(fmsg(f"init_w_R2D1 [Vision backbone]: "
                              f"{args['model_params']['init_w_R2D1']}"))

        if 'I3D' in args['model_params']['l_vision_backbones']:
            DLLogger.log(fmsg(f"init_w_I3D [Vision backbone]: "
                              f"{args['model_params']['init_w_I3D']}"))
        if 'ResNet18' in args['model_params']['l_audio_backbones']:
            DLLogger.log(fmsg(f"init_w_ResNet18 [Audio backbone]: "
                              f"{args['model_params']['init_w_ResNet18']}"))

        DLLogger.log(fmsg(f"Goal: {args['goal']}"))
        DLLogger.log(fmsg(f"Join modalities: "
                          f"{args['model_params']['joint_modalities']}"))
        DLLogger.log(fmsg(f"Intra modality fusion: "
                          f"{args['model_params']['intra_modal_fusion']}"))

        output_format = args['model_params']['output_format']
        DLLogger.log(fmsg(f"Output format: {output_format}"))

    elif mode == 'Eval':
        parser.add_argument("--eval_set", type=str,
                            help="Evaluation set: test, valid, train")
        parser.add_argument("--fd_exp", type=str,
                            help="Absolute path to the exp folder")
        input_args, _ = parser.parse_known_args()
        eval_set = input_args.eval_set
        fd_exp = input_args.fd_exp
        assert eval_set in ['test', 'valid', 'train'], eval_set
        assert os.path.isdir(fd_exp), fd_exp

        store_results_pkl = join(fd_exp, f'{eval_set}-reevaluation.pkl')
        if os.path.isfile(store_results_pkl):
            print(f"This evaluation has already been done. Exiting."
                  f"Fd_exp: {fd_exp}."
                  f"Eval_set: {eval_set}")
            sys.exit(0)

        args_path = join(fd_exp, 'final_config.yml')
        assert os.path.isfile(args_path), args_path
        with open(args_path, 'r') as fx:
            args: dict = yaml.safe_load(fx)

        # fig a possible glitch: somehow, the samples may be loaded by
        # dataloader in unordered way. to limit this, we set the batch size
        # to 1.
        new_bs = None

        if eval_set == 'test':
            new_bs = 1
            args['test_params']['loader_params']['batch_size'] = new_bs

        elif eval_set == 'valid':
            new_bs = 1
            args['val_params']['loader_params']['batch_size'] = new_bs

        elif eval_set == 'train':
            new_bs = 1
            args['train_params']['loader_params']['batch_size'] = new_bs

        else:
            raise NotImplementedError(eval_set)

        # ======================================================================
        l_vision_backbones = args['model_params'][
            'l_vision_backbones'].split('+')
        assert len(l_vision_backbones) == len(set(l_vision_backbones))
        for bk in l_vision_backbones:
            assert bk in ['R2D1', 'I3D', 'None'], bk

        if 'None' in l_vision_backbones:
            assert len(l_vision_backbones) == 1, len(l_vision_backbones)

        if l_vision_backbones == ['None']:
            l_vision_backbones = []

        args['model_params']['l_vision_backbones']: list = l_vision_backbones

        l_audio_backbones = args['model_params']['l_audio_backbones'].split('+')
        assert len(l_audio_backbones) == len(set(l_audio_backbones))
        for bk in l_audio_backbones:
            assert bk in ['ResNet18', 'wavLM', 'None'], bk

        if 'None' in l_audio_backbones:
            assert len(l_audio_backbones) == 1, len(l_audio_backbones)

        if l_audio_backbones == ['None']:
            l_audio_backbones = []

        args['model_params']['l_audio_backbones']: list = l_audio_backbones

        n_backbones = len(l_vision_backbones) + len(l_audio_backbones)

        assert args['goal'] in ['PRETRAINING', 'TRAINING'], args['goal']

        intra_modal_fusion = args['model_params']['intra_modal_fusion']
        assert intra_modal_fusion in ['None', 'feat_concat_fc',
                                      'encoder_plus_self_attention'], intra_modal_fusion

        if intra_modal_fusion in ['encoder_plus_self_attention',
                                  'feat_concat_fc']:
            assert any(
                [len(l_audio_backbones) == 2, len(l_vision_backbones) == 2])
            if len(l_audio_backbones) == 2:
                assert 'ResNet18' in l_audio_backbones
                assert 'wavLM' in l_audio_backbones

            if len(l_vision_backbones) == 2:
                assert 'R2D1' in l_vision_backbones
                assert 'I3D' in l_vision_backbones

        joint_modalities = args['model_params']['joint_modalities']

        if args['goal'] == 'PRETRAINING':
            assert n_backbones == 1, n_backbones  # pretrain one backbone at once.
            assert intra_modal_fusion == 'None', intra_modal_fusion
            assert joint_modalities == 'NONE', joint_modalities


        elif args['goal'] == 'TRAINING':
            assert len(l_vision_backbones) > 0, len(l_vision_backbones)
            assert len(l_audio_backbones) > 0, len(l_audio_backbones)
            assert n_backbones > 1, n_backbones

            assert joint_modalities in ['NONE', 'TRANSFORMER', 'FC'], \
                joint_modalities

        else:
            raise NotImplementedError(args['goal'])

        output_format = args['model_params']['output_format']
        assert output_format in ['FC', 'SELF_ATTEN'], output_format

        assert args['split'] in ['DEFAULT',
                                 'ROUND1',
                                 'ROUND2',
                                 'ROUND3',
                                 'ROUND4',
                                 'ROUND5'], args['split']

        args = reset_data_paths_to_default(args)

        args = auto_set_tr_vl_tst_paths(args)


        freeze_vision_R2D1 = args['model_params']['freeze_vision_R2D1']
        assert isinstance(freeze_vision_R2D1, bool), type(freeze_vision_R2D1)
        if not freeze_vision_R2D1:
            assert 'R2D1' in l_vision_backbones

        freeze_vision_I3D = args['model_params']['freeze_vision_I3D']
        assert isinstance(freeze_vision_I3D, bool), type(freeze_vision_I3D)
        if not freeze_vision_I3D:
            assert 'I3D' in l_vision_backbones

        freeze_audio_ResNet18 = args['model_params']['freeze_audio_ResNet18']
        assert isinstance(freeze_audio_ResNet18, bool), type(
            freeze_audio_ResNet18)
        if not freeze_audio_ResNet18:
            assert 'ResNet18' in l_audio_backbones

        R2D1_ft_dim_reduce = args['model_params']['R2D1_ft_dim_reduce']
        assert R2D1_ft_dim_reduce in ['FLATTEN', 'MAX',
                                      'AVG'], R2D1_ft_dim_reduce

        init_w_R2D1 = args['model_params']['init_w_R2D1']
        assert init_w_R2D1 in ['RANDOM', 'KINETICS400', 'AFFWILD2',
                               'OUR_AFFWILD2'], init_w_R2D1

        init_w_I3D = args['model_params']['init_w_I3D']
        assert init_w_I3D in ['RANDOM', 'KINETICS400', 'AFFWILD2',
                              'OUR_AFFWILD2'], init_w_I3D

        init_w_ResNet18 = args['model_params']['init_w_ResNet18']
        assert init_w_ResNet18 in ['RANDOM', 'IMAGENET', 'AFFWILD2',
                                   'OUR_AFFWILD2'], init_w_ResNet18
        # ======================================================================

        log_backends = [
            ArbJSONStreamBackend(Verbosity.VERBOSE,
                                 join(fd_exp, f"log-eval-{eval_set}.json")),
            ArbTextStreamBackend(Verbosity.VERBOSE,
                                 join(fd_exp, f"log-eval-{eval_set}.txt")),
        ]

        if args['verbose']:
            log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))

        DLLogger.init_arb(backends=log_backends, master_pid=os.getpid())

        DLLogger.log(fmsg("Start time: {}".format(args['t0'])))

        DLLogger.log(fmsg(f"Fold: {args['split']} -  EVAL: {eval_set}"))

        if new_bs is not None:
            DLLogger.log(f"We have reset the batch size to {new_bs} for "
                         f"the set: {eval_set} to avoid a bug.")

        eval_config = {
            'eval_set': eval_set,
            'fd_exp': fd_exp
        }


    DLLogger.flush()
    return args, mode, eval_config
