#!/usr/bin/env python
import sys
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
from fitness import dmi_effect, dmi_effect_recessive, dmi_effect_dominant, heterosis_effect, hybrid_fitness
from fixation_probabilities import estimate_alpha
from mpl_axes_aligner import align
from matplotlib.ticker import MultipleLocator


def plot_regimes(params, s, recessive, dominant, output):
    """
    Plot the regimes. Saves figure {output}/regimes.pdf
    :param params: list of tuples, initial parameters for strength of heterosis and DMI effects
    :param s: float, selection coefficient
    :param recessive: boolean, if to assume DMIs due to recessive-dominant interactions
    :param dominant: boolean, if to assume DMIs due to dominant-dominant interactions
    :param output: str, output directory
    """
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.87, 7.87))
    plt.subplots_adjust(wspace=0.2, hspace=0.25)
    t = np.arange(20)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    n_plot = 0
    secondary_axes = dict()
    for i, param in enumerate(params):
        h = param[0]
        d = param[1]
        alpha = estimate_alpha(h, d, recessive, dominant)
        print('\u03B7_1=' + str(round(h, 3)) + ', \u03B4_1=' + str(round(d, 3)) + ': \u03B1=' + str(round(alpha, 3)))

        if recessive:
            dmi = dmi_effect_recessive(t, d)
        elif dominant:
            dmi = dmi_effect_dominant(t, d)
        else:
            dmi = dmi_effect(t, d)
        het = heterosis_effect(t, h)
        selection = np.full_like(het, s)
        selection[0] = 0.0
        w_ab, w_bb = hybrid_fitness(selection, het, dmi)
        secondary_axes[i] = ax[i // 2, i % 2].twinx()
        secondary_axes[i].plot(t, w_ab, label=r"$w_{H_{AB}, t}$", color='royalblue', marker='o', ls=':', markersize=4,
                               alpha=0.7)
        if i % 2 == 1:
            secondary_axes[i].tick_params(axis='y', labelcolor='royalblue', labelsize=9)
            secondary_axes[i].set_ylabel(r'$w_{H_{AB}, t}$', color='royalblue', fontdict={'fontsize': 10},)
        else:
            secondary_axes[i].set_yticklabels([])
        ax[i // 2, i % 2].scatter(t, dmi, label='DMI', color='grey', marker='v', s=17)
        ax[i // 2, i % 2].scatter(t, het, label='Heterosis', color='grey', marker='x', s=17)
        ax[i // 2, i % 2].scatter(t, selection, color='black', label='intrinsic selection', marker='o', s=17)
        ax[i // 2, i % 2].set_xticks(np.arange(1, 15 + 2, 2))
        ax[i // 2, i % 2].xaxis.set_minor_locator(MultipleLocator())
        ax[i // 2, i % 2].set_xlim([0.85, 15])
        if i // 2 == 0 and i % 2 == 0:
            title_str = 'Classical model\n'
        elif i // 2 == 0 and i % 2 == 1:
            title_str = 'Strong heterosis\n'
        elif i // 2 == 1 and i % 2 == 0:
            title_str = 'Strong DMI effects\n'
        elif i // 2 == 1 and i % 2 == 1:
            title_str = 'Strong heterosis and DMI effects\n'
        if recessive:
            title_str += r'($\eta_1=$' + str(round(h, 2)) + r', $\delta_{rd, 2}=$' + str(round(d, 2)) + r'; $\alpha=$' + str(round(alpha, 3)) + ")"
        elif dominant:
            title_str += r'($\eta_1=$' + str(round(h, 2)) + r', $\delta_{dd, 1}=$' + str(round(d, 2)) + r'; $\alpha=$' + str(round(alpha, 3)) + ")"
        else:
            title_str += r'($\eta_1=$' + str(round(h, 2)) + r', $\delta_{1}=$' + str(round(d, 2)) + r'; $\alpha=$' + str(round(alpha, 3)) + ")"
        ax[i // 2, i % 2].set_title(title_str, fontsize=10)
        ax[i // 2, i % 2].text(-.1, 1.05, str(alphabet[n_plot]), transform=ax[i // 2, i % 2].transAxes,
                               weight='bold', fontsize=12)
        secondary_axes[i].set_yticks(np.linspace(0, 2, 11))

        n_plot += 1

    secondary_axes[0].get_shared_y_axes().join(*secondary_axes.values())
    org1 = 0.0
    org2 = 0.99
    pos = 0.5
    align.yaxes(ax[0, 0], org1, secondary_axes[0], org2, pos)

    ax[1, 1].legend(bbox_to_anchor=(-0.15, -.17), ncol=4, loc='upper center')
    ax[1, 1].set_xlabel('t, in generations', fontdict={'fontsize': 10})
    ax[1, 0].set_xlabel('t, in generations', fontdict={'fontsize': 10})
    ax[0, 0].set_ylabel("Effect on fitness", fontdict={'fontsize': 10}, labelpad=2)
    ax[1, 0].set_ylabel("Effect on fitness", fontdict={'fontsize': 10}, labelpad=2)
    ax[1, 0].tick_params(axis='x', labelsize=9)
    ax[1, 1].tick_params(axis='x', labelsize=9)
    ax[0, 0].tick_params(axis='y', labelsize=9)
    ax[1, 0].tick_params(axis='y', labelsize=9)
    if recessive:
        fig.savefig("{}regimes_recessive.pdf".format(output), dpi=600, bbox_inches='tight')
    elif dominant:
        fig.savefig("{}regimes_dominant.pdf".format(output), dpi=600, bbox_inches='tight')
    else:
        fig.savefig("{}regimes.pdf".format(output), dpi=600, bbox_inches='tight')
    plt.close()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', help='Parameter space for initial strength of heterosis and DMI effects.'
                                               'Alternate initial value for heterosis and DMI effects,'
                                               'separated by a space. In total 4 pairs are required'
                                               'Default=[0.0, 0.0, 0.8, 0.0, 0.0, -0.8, 0.8, -0.8]',
                        default=[0.0, 0.0, 0.8, 0.0, 0.0, -0.8, 0.8, -0.8], nargs=8, action='store', type=float)
    parser.add_argument('-s', '--selection_coefficient', help='Selection coefficient, default: s=0.1', default=0.1,
                        type=float)
    parser.add_argument('-r', '--recessive', help='DMIs due to recessive-dominant epistasis; default=False',
                        default=False, action='store_true')
    parser.add_argument('-d', '--dominant', help='DMIs due to dominant-dominant epistasis; default=False',
                        default=False, action='store_true')
    parser.add_argument('-o', '--output', help='Output directory. Will create a regimes.pdf file in this directory' +
                        ', default=./plots', default='./plots/')
    args = parser.parse_args()
    if len(args.params) != 8:
        raise AssertionError("4 pairs of initial values for heterosis and DMI effects are required")
    params = [(h, d) for h, d in zip(args.params[::2], args.params[1::2])]
    recessive = args.recessive
    dominant = args.dominant
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    if not args.output.endswith('/'):
        output = args.output + '/'
    else:
        output = args.output
    if recessive and dominant:
        raise AssertionError("Set either -r or -d flag, not both.")
    plot_regimes(params, args.selection_coefficient, recessive, dominant, output)


if __name__ == '__main__':
    main(sys.argv[1:])
