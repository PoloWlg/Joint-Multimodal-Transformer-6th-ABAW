import os
from os.path import join

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

_TRAIN = 'train'
_VALID = 'valid'
_VALENCE = 'valence'
_AROUSAL = 'arousal'
_AVG = 'avg_valence_arousal'


class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def state_dict_to_cpu(state_dict):
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()

    return state_dict


def state_dict_to_gpu(state_dict, device=None):
    for k, v in state_dict.items():
        if device is None:
            v = v.cuda()
        else:
            v = v.to(device)

        state_dict[k] = v

    return state_dict


def fmsg(msg, upper=True):
    """
    Format message.
    :param msg:
    :param upper:
    :return:
    """
    if upper:
        msg = msg.upper()
    n = min(120, max(80, len(msg)))
    top = "\n" + "=" * n
    middle = " " * (int(n / 2) - int(len(msg) / 2)) + " {}".format(msg)
    bottom = "=" * n + "\n"

    output_msg = "\n".join([top, middle, bottom])
    return output_msg


def plot_tracker(tracker: dict,
                 fdout: str,
                 dpi=300
                 ):
    epochs = list(range(len(tracker[_TRAIN][_VALENCE])))
    best_idx = tracker[_TRAIN]['best_idx']


    lambda_vals = np.asarray(epochs)

    font_sz = 7
    shift = 5 / 100.  # how much to shift ticks top and bottom when
    # plotting values.

    # fig, ax1 = plt.subplots()
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 2.5))
    start = 0
    z = lambda_vals.size
    x = list(range(start, z + start, 1))

    # Valid
    linewidth = 1.

    color_1 = 'tab:red'
    color_2 = 'tab:blue'
    lns1 = ax1.plot(x,
                    tracker[_VALID][_VALENCE],
                    color=color_1,
                    label='Valid / Valence',
                    linewidth=linewidth
                    )
    lns2 = ax1.plot(x,
                    tracker[_VALID][_AROUSAL],
                    color=color_2,
                    label='Valid / Arousal',
                    linewidth=linewidth
                    )

    # Train
    lns3 = ax1.plot(x,
                    tracker[_TRAIN][_VALENCE],
                    color=color_1,
                    label='Train / Valence',
                    linestyle='dashed',
                    linewidth=linewidth / 2,
                    alpha=0.2
                    )

    lns4 = ax1.plot(x,
                    tracker[_TRAIN][_AROUSAL],
                    color=color_2,
                    label='Train / Arousal',
                    linestyle='dashed',
                    linewidth=linewidth / 2.,
                    alpha=0.2
                    )

    for subset in tracker:
        for k in [_AROUSAL, _VALENCE]:
            ax1.plot([best_idx], [tracker[subset][k][best_idx]],
                     marker='o',
                     markersize=2,
                     color="red"
                     )


    ax1.set_xlabel('Epochs', fontsize=font_sz)
    ax1.set_ylabel('Valence, Arousal', fontsize=font_sz)

    plt.title('Train / valid performance', fontsize=font_sz)

    xticks = []
    for v_ in epochs:
        if v_ < 1:
            xticks.append(str(v_))
        else:
            xticks.append(str(int(v_)))


    ax1.set_xticks(x, xticks)
    ax1.xaxis.set_tick_params(labelsize=4)
    ax1.grid(b=True, which='major', linestyle='-', alpha=0.2)
    ax1.tick_params(axis='y', labelsize=font_sz)

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower center', prop={'size': 6, 'weight':'bold'})
    os.makedirs(fdout, exist_ok=True)
    fig.savefig(join(fdout, 'tracker.png'), bbox_inches='tight', dpi=dpi)
    plt.close('all')

    del fig
