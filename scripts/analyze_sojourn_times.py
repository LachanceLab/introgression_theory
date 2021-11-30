#!/usr/bin/python
import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from wright_fisher_models import wright_fisher, standard_wright_fisher


def compare_sojourn_times(p0, nr_simulations, params, recessive, dominant, pars, output):
    """
    Compare sojourn times under our model with sojourn times under classical model.
    If both recessive and dominance are False the semidominance-dominance epistasis model is assumed for DMI effects
    :param p0: float, initial allele frequency
    :param nr_simulations: int, number of simulations to run
    :param params: list, list of tuples with initial values for heterosis and DMI effects
    :param recessive: boolean, whether to assume recessive-dominance epistasis model for DMI effects
    :param dominant: boolean, whether to assume dominance-dominance epistasis model for DMI effects
    :param pars: dict, parameters for WF model
    :param output: str, output directory
    """
    fig = plt.figure(figsize=(7.87, 7.87))
    grid = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 0.2, 1, 1])
    grid.update(wspace=0.3, hspace=0.35)

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    n_plot = 0
    handles = [Line2D([0], [0], color='cornflowerblue', ls='-'),
               Line2D([0], [0], color='dimgrey', ls='-'),
               Line2D([0], [0], color='cornflowerblue', ls='--'),
               Line2D([0], [0], color='dimgrey', ls='--')]
    labels = ['Extinction time adjusted model', 'Extinction time classical model',
              'Fixation time adjusted model', 'Fixation time classical model']

    for i, param in enumerate(params):
        pars['h'] = param[0]
        pars['d'] = param[1]
        extinction_time = []
        fixation_time = []
        sojourn_time = []
        expected_extinction_time = []
        expected_fixation_time = []
        expected_sojourn_time = []
        for y in range(nr_simulations):
            frequency = wright_fisher(p0, pars, recessive, dominant)
            if frequency[-1] == 1.0:
                fixation_time.append(len(frequency) - 1)
            else:
                extinction_time.append(len(frequency) - 1)
            sojourn_time.append(len(frequency) - 1)
            freq_mutants = standard_wright_fisher(p0, pars['s'], pars['N'])
            if freq_mutants[-1] == 1.0:
                expected_fixation_time.append(len(freq_mutants) - 1)
            else:
                expected_extinction_time.append(len(freq_mutants) - 1)
            expected_sojourn_time.append(len(freq_mutants) - 1)
        if i % 2 == 0:
            col1 = 0
            col2 = 1
        elif i % 2 == 1:
            col1 = 3
            col2 = 4

        ax = fig.add_subplot(grid[i // 2, col1])
        ax1 = fig.add_subplot(grid[i // 2, col2])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.yaxis.set_visible(False)
        ax.text(-25, 1.1, str(alphabet[n_plot]), fontsize=12, weight='bold')

        ax.set_xlim([0, 80])
        ax1.set_xlim([900, 1600])
        if i // 2 == 0 and i % 2 == 0:
            title_str = 'Classical model\n'
        elif i // 2 == 0 and i % 2 == 1:
            title_str = 'Strong heterosis\n'
        elif i // 2 == 1 and i % 2 == 0:
            title_str = 'Strong DMI effects\n'
        elif i // 2 == 1 and i % 2 == 1:
            title_str = 'Strong heterosis and DMI effects\n'
        if recessive:
            title_str += r'($\eta_1=$' + str(round(pars['h'], 2)) + r', $\delta_{rd, 2}=$' + str(round(pars['d'], 2)) +\
                         ')'
        elif dominant:
            title_str += r'($\eta_1=$' + str(round(pars['h'], 2)) + r', $\delta_{dd, 1}=$' + str(round(pars['d'], 2)) +\
                         ')'
        else:
            title_str += r'($\eta_1=$' + str(round(pars['h'], 2)) + r', $\delta_{sd, 1}=$' + str(round(pars['d'], 2)) +\
                         ')'
        ax.set_title(title_str, x=1.3, fontsize=10)
        kwargs = dict(transform=ax.transAxes, clip_on=False)
        ax.hist(expected_extinction_time, weights=np.full(len(expected_extinction_time),
                                                          1 / len(expected_extinction_time)),
                histtype='step', color='dimgrey', bins=102, cumulative=True, range=(0, 102))
        ax.hist(extinction_time, weights=np.full(len(extinction_time),
                                                 1 / len(extinction_time)),
                histtype='step', color='cornflowerblue', bins=102, cumulative=True, range=(0, 102), )
        if len(expected_fixation_time) > 0:
            ax1.hist(expected_fixation_time, weights=np.full(len(expected_fixation_time),
                                                             1 / len(expected_fixation_time)),
                     histtype='step', color='dimgrey', bins=750, cumulative=True, range=(900, 1650), ls='--')

        if len(fixation_time) > 0:
            ax1.hist(fixation_time, weights=np.full(len(fixation_time), 1 / len(fixation_time)),
                     histtype='step', color='cornflowerblue', bins=750, cumulative=True, range=(900, 1650), ls='--')
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        d = 0.03
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax1.transAxes)  # switch to the bottom axes
        ax1.plot((-d, +d), (-d, +d), **kwargs)
        if i // 2 == 1:
            ax.set_xlabel('Sojourn time, in generations', fontdict={'fontsize': 10})
            ax.xaxis.set_label_coords(1.1, -0.12)
        if i // 2 == 1 and i % 2 == 0:
            ax.legend(handles=handles, labels=labels, bbox_to_anchor=(2.5, -.2), loc='upper center', ncol=2)
        if i % 2 == 0:
            ax.set_ylabel('Cumulative probability of loss/fixation', fontdict={'fontsize': 10})
        n_plot += 1
    if recessive:
        fig.savefig("{}sojourn_times_recessive.pdf".format(output), dpi=600, bbox_inches='tight')
    elif dominant:
        fig.savefig("{}sojourn_times_dominant.pdf".format(output), dpi=600, bbox_inches='tight')
    else:
        fig.savefig("{}sojourn_times.pdf".format(output), dpi=600, bbox_inches='tight')


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', help='Parameter space for initial strength of heterosis and DMI effects.'
                                               'Alternate initial value for heterosis and DMI effects,'
                                               'separated by a space. In total 4 pairs are required'
                                               'Default=[0.0, 0.0, 0.8, 0.0, 0.0, -0.8, 0.8, -0.8]',
                        default=[0.0, 0.0, 0.8, 0.0, 0.0, -0.8, 0.8, -0.8], nargs=8, action='store', type=float)
    parser.add_argument('-N', '--population_size', help='Population size, default=10000', type=int, default=10000)
    parser.add_argument('-s', '--selection_coefficient', help='Selection coefficient, default: s=0.01', default=0.01,
                        type=float)
    parser.add_argument('-n', '--number_simulations', help='Number of simulations to run, default=10000',
                        default=100000, type=int)
    parser.add_argument('-r', '--recessive', help='DMIs due to recessive-dominant epistasis; default=False',
                        default=False, action='store_true')
    parser.add_argument('-d', '--dominant', help='DMIs due to dominant-dominant epistasis; default=False',
                        default=False, action='store_true')
    parser.add_argument('-o', '--output', help='Output directory. Will create a sojourn_times.pdf file in this ' +
                                               'directory, default=./plots', default='./plots/')
    args = parser.parse_args()
    pars = dict()
    pars['s'] = args.selection_coefficient
    pars['N'] = args.population_size
    nr_simulations = args.number_simulations
    p0 = 1 / (2 * pars['N'])
    if len(args.params) != 8:
        raise AssertionError("4 pairs of initial values for heterosis and DMI effects are required")
    params = [(h, d) for h, d in zip(args.params[::2], args.params[1::2])]
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    if not args.output.endswith('/'):
        output = args.output + '/'
    else:
        output = args.output
    if args.recessive and args.dominant:
        raise AssertionError("Set either -r or -d flag, not both.")
    compare_sojourn_times(p0, nr_simulations, params, args.recessive, args.dominant, pars, output)


if __name__ == '__main__':
    main(sys.argv[1:])
