from typing import Iterable
import copy

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import my_lr_scheduler


class Dict2Obj(object):
    """
    Convert a dictionary into a class where its attributes are the keys of
     the dictionary, and
    the values of the attributes are the values of the keys.
    """
    def __init__(self, dictionary):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s" % attrs


def build_optimizer_for_params(params: Iterable[object], hparams: object):
    """
    Builds an optimizer for the given params, and their hyper-paramerters.
    """

    if hparams.name_optimizer == 'sgd':
        optimizer = SGD(params=params,
                        momentum=hparams.momentum,
                        dampening=hparams.dampening,
                        weight_decay=hparams.weight_decay,
                        nesterov=hparams.nesterov
                        )

    elif hparams.name_optimizer == 'adam':
        optimizer = Adam(params=params,
                         betas=(hparams.beta1, hparams.beta2),
                         eps=hparams.eps_adam,
                         weight_decay=hparams.weight_decay,
                         amsgrad=hparams.amsgrad
                         )

    else:
        raise ValueError(f"Unsupported optimizer name "
                         f"`{hparams.name_optimizer}` ... [NOT OK]")

    if hparams.lr_scheduler:
        if hparams.name_lr_scheduler == 'step':
            lrate_scheduler = lr_scheduler.StepLR(optimizer,
                                                  step_size=hparams.step_size,
                                                  gamma=hparams.gamma,
                                                  last_epoch=hparams.last_epoch
                                                  )


        elif hparams.name_lr_scheduler == 'cosine':
            lrate_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=hparams.t_max,
                eta_min=hparams.min_lr,
                last_epoch=hparams.last_epoch
            )

        elif hparams.name_lr_scheduler == 'mystep':
            lrate_scheduler = my_lr_scheduler.MyStepLR(
                optimizer,
                step_size=hparams.step_size,
                gamma=hparams.gamma,
                last_epoch=hparams.last_epoch,
                min_lr=hparams.min_lr)


        elif hparams.name_lr_scheduler == 'mycosine':
            lrate_scheduler = my_lr_scheduler.MyCosineLR(
                optimizer,
                coef=hparams.coef,
                max_epochs=hparams.max_epochs,
                min_lr=hparams.min_lr,
                last_epoch=hparams.last_epoch)


        elif hparams.name_lr_scheduler == 'multistep':
            lrate_scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=hparams.milestones,
                gamma=hparams.gamma,
                last_epoch=hparams.last_epoch)

        elif hparams.name_lr_scheduler == 'reduce_on_plateau':
            lrate_scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=hparams.mode,
                factor=hparams.factor,
                patience=hparams.patience,
                min_lr=hparams.min_lr
            )

        else:
            raise ValueError(f"Unsupported LR scheduler "
                             f"`{hparams.name_lr_scheduler}` ... [NOT OK]")
    else:
        lrate_scheduler = None


    return optimizer, lrate_scheduler


def standardize_optimizers_hparams(optm_dict: dict, initial: str):
    """
    Standardize the keys of a dict for the optimizer.
    all the keys starts with 'initial__key' where we keep only the key and
    delete the initial.
    the dict should not have a key that has a dict as value. we do not deal
    with this case. an error will be raise.

    :param optm_dict: dict with specific keys.
    :return: a copy of optm_dict with standardized keys.
    """
    assert isinstance(optm_dict, dict), type(optm_dict)
    new_optm_dict = copy.deepcopy(optm_dict)
    loldkeys = list(new_optm_dict.keys())

    for k in loldkeys:
        if k.startswith(initial):
            msg = f"'{k}' is a dict. it must not be the case." \
                  "otherwise, we have to do a recursive thing...."
            assert not isinstance(new_optm_dict[k], dict), msg

            new_k = k.split('__')[1]
            assert new_k not in new_optm_dict, new_k
            new_optm_dict[new_k] = new_optm_dict.pop(k)

    return new_optm_dict


def get_optimizer_for_params(args_holder: dict,
                             params: Iterable[object],
                             initial: str
                             ):
    """
    Get optimizer for a set of parameters. Hyper-parameters are in args_holder.
    :param args_holder: dict containing hyper-parameters.
    :param params: list of params.
    :param initial: str. initial to extract the hyper-parameters.
    :return:
    """
    hparams = copy.deepcopy(args_holder)
    hparams = standardize_optimizers_hparams(hparams, initial)
    hparams = Dict2Obj(hparams)

    _params = [
        {'params': params, 'lr': hparams.lr}
    ]

    optimizer, lrate_scheduler = build_optimizer_for_params(_params, hparams)

    return optimizer, lrate_scheduler

