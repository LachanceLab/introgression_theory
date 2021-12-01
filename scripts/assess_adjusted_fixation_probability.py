#!/usr/bin/python
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
from fixation_probabilities import adjusted_probability_of_fixation, estimate_alpha
from wright_fisher_models import wright_fisher, wright_fisher_fixation_prob
import seaborn as sns
from itertools import islice, takewhile, repeat
import multiprocessing as mp


def assess_fixation_probability_helper(args):
    """
    Helper function for multiprocessing the assessment of the fixation probability formula.
    Runs the actual simulations over the parameter space (heterosis, DMIs)
    @param args: list of tuples, Input variables. See plot_fixation_probabilities for the meaning of the parameters
    @return: (np.array, np.array), empirical and theoretical fixation probability (val_range.shape[0])
    """
    vals, pars, p0, recessive, dominant, nr_simulations, val_range, heterosis, dmis = args
    emp_fix = np.zeros_like(val_range)
    theo_fix = np.zeros_like(val_range)
    for val in vals:
        if heterosis:
            pars['h'] = val
        elif dmis:
            pars['d'] = val
        if pars['s'] == 0:
            fixations = 0
            for y in range(nr_simulations):
                frequency = wright_fisher(p0, pars, recessive, dominant)
                if frequency[-1] == 1.0:
                    fixations += 1
            emp_p_fixation = fixations / nr_simulations
        else:
            emp_p_fixation = wright_fisher_fixation_prob(int(p0 * 2 * pars['N']), pars, recessive, dominant, nr_simulations)
        emp_fix[np.where(val_range == val)[0]] = emp_p_fixation
        alpha = estimate_alpha(pars['h'], pars['d'], recessive, dominant)
        theo_p_fixation = adjusted_probability_of_fixation(pars['N'], pars['s'], p0, alpha)
        theo_fix[np.where(val_range == val)[0]] = theo_p_fixation
    return emp_fix, theo_fix


def plot_fixation_probabilities(p0, nr_simulations, values_s, recessive, dominant, pars, output, threads):
    """
    Plot the empirical and theoretical fixation probability for different strengths of heterosis and DMI effects.
    If both recessive and dominant are false the semidominance-dominance epistasis model is assumed for DMI effects.
    :param p0: float, initial allele frequency
    :param nr_simulations: int, number of simulations to run
    :param values_s: list, values for s to use
    :param recessive: boolean, whether to assume recessive-dominance epistasis model for DMI effects
    :param dominant: boolean, whether to assume dominant recessive-dominance epistasis model for DMI effects
    :param pars: dict, parameters for WF model
    :param output: str, output dir
    :param threads: int, number of CPUs
    """
    heterosis_effects = pars['h']
    fig, ax = plt.subplots(1, 2, figsize=(7.87, 5), sharey=True)
    marker_symbols = ['o', 'x', '^', 's', 'p', '*']
    colors = ["blue", "cornflowerblue", "deepskyblue", "lightgrey", "grey", 'darkslategrey']
    h_range = np.arange(0., 1.025, 0.025)
    print("Heterosis:")
    # run simulations
    for s, marker, c in zip(values_s, marker_symbols, colors):
        pars['s'] = s
        iterator = iter(h_range)
        chunks = takewhile(bool, (list(islice(iterator, 1)) for _ in repeat(None)))
        ready_to_map = [(h_val, pars, p0, recessive, dominant, nr_simulations, h_range,
                         True, False) for h_val in chunks]
        # multiprocess --> run different parameter combinations in parallel
        pool = mp.Pool(processes=threads)
        results = pool.map(assess_fixation_probability_helper, ready_to_map)
        # combine results
        emp_fix = sum([result[0] for result in results])
        theo_fix = sum([result[1] for result in results])
        pool.close()
        pool.join()
        # statistic, pvalue = kstest(emp_fix, theo_fix)
        # print('s={}, p={}'.format(s, pvalue))
        ax[0].plot(np.arange(0., 1.025, 0.025), theo_fix, color=c)
        ax[0].scatter(np.arange(0., 1.025, 0.025)[::4], emp_fix[::4], marker=marker, label='s={:.3f}'.format(s),
                      color=c)
    ax[0].text(-0.1, 1.05, 'A', transform=ax[0].transAxes, fontsize=12, weight='bold')
    ax[0].set_xlabel(r'Heterosis effects ($\eta_1$)')

    ax[0].set_ylabel("Probability of fixation")
    # add dummy to create a second row
    handles, labels = ax[0].get_legend_handles_labels()
    handles.insert(1, Line2D([], [], marker='o', markerfacecolor='none', markeredgecolor='none', ls=''))
    labels.insert(1, '')
    ax[0].legend(handles, labels, loc='upper center', bbox_to_anchor=(1.05, -.14), ncol=len(values_s), labelspacing=1,
                 columnspacing=1, handletextpad=0.5)

    pars['h'] = heterosis_effects
    d_range = np.arange(-1.0, 0.025, 0.025)
    d_range[-1] = 0.0
    # run simulations
    print("DMIs")
    for s, marker, c in zip(values_s, marker_symbols, colors):
        pars['s'] = s
        iterator = iter(d_range)
        chunks = takewhile(bool, (list(islice(iterator, 1)) for _ in repeat(None)))
        ready_to_map = [(d_val, pars, p0, recessive, dominant, nr_simulations, d_range,
                         False, True) for d_val in chunks]
        # multiprocess --> run different parameter combinations in parallel
        pool = mp.Pool(processes=threads)
        results = pool.map(assess_fixation_probability_helper, ready_to_map)
        emp_fix = sum([result[0] for result in results])
        theo_fix = sum([result[1] for result in results])
        pool.close()
        pool.join()
        # statistic, pvalue = kstest(emp_fix, theo_fix)
        # print('s={}, p={}'.format(s, pvalue))
        ax[1].plot(d_range, theo_fix, color=c)
        ax[1].scatter(d_range[::4], emp_fix[::4], marker=marker, color=c)
    ax[1].text(-0.1, 1.05, 'B', transform=ax[1].transAxes, fontsize=12, weight='bold')
    if recessive:
        ax[1].set_xlabel(r'DMI effects ($\delta_{rd, 2}$)')
    elif dominant:
        ax[1].set_xlabel(r'DMI effects ($\delta_{dd, 1}$)')
    else:
        ax[1].set_xlabel(r'DMI effects ($\delta_{sd, 1}$)')

    handles = [Line2D([0], [0], ls='-', color='black'), Line2D([0], [0], ls='', color='black', marker='o')]
    labels = ['theoretical', 'simulations']
    ax[1].legend(handles, labels, loc='upper center', bbox_to_anchor=(-0.10, -.21), ncol=2, frameon=False,
                 columnspacing=1, handletextpad=0.5)
    if recessive:
        fig.savefig('{}adjusted_pfix_recessive.pdf'.format(output), dpi=600, bbox_inches='tight')
    elif dominant:
        fig.savefig('{}adjusted_pfix_dominant.pdf'.format(output), dpi=600, bbox_inches='tight')
    else:
        fig.savefig('{}adjusted_pfix.pdf'.format(output), dpi=600, bbox_inches='tight')
    plt.close()


def get_error_helper(args):
    """
    Helper function for parallelizing error assessment
    @param args: list of tuples, [(chunks, pars, nr_simulations, p0, h_range, d_range, recessive, dominant)]
                                see explore_full_parameter_space for parameter meaning
    @return: np.array, (h_range.shape[0], d_range.shape[0]) errors
    """
    chunks, pars, nr_simulations, p0, h_range, d_range, recessive, dominant = args
    error = np.zeros((h_range.shape[0], d_range.shape[0]))
    for chunk in chunks:
        pars['h'], pars['d'] = chunk
        # fixations = 0
        # for y in range(nr_simulations):
        #     frequency = wright_fisher(p0, pars, recessive, dominant)
        #     if frequency[-1] == 1.0:
        #         fixations += 1
        # emp_p_fixation = fixations / nr_simulations
        emp_p_fixation = wright_fisher_fixation_prob(int(p0 * 2 * pars['N']), pars, recessive, dominant, nr_simulations)
        alpha = estimate_alpha(pars['h'], pars['d'], recessive, dominant)
        theo_p_fixation = adjusted_probability_of_fixation(pars['N'], pars['s'], p0, alpha)
        error[np.where(pars['h'] == h_range)[0], np.where(pars['d'] == d_range)[0]] = emp_p_fixation - theo_p_fixation
    return error


def explore_full_parameter_space(p0, nr_simulations, recessive, dominant, pars, output, threads):
    """
    Compute difference of empirical fixation probability and adjusted theoretical fixation probability
    for full parameter space. If both recessive and dominant are false the semidominance-dominance
    epistasis model is assumed for DMI effects.
    :param p0: float, initial allele frequency
    :param nr_simulations: int, number of simulations to run
    :param recessive: boolean, whether to assume recessive-dominance epistasis model for DMI effects
    :param dominant: boolean, whether to assume dominant recessive-dominance epistasis model for DMI effects
    :param pars: dict, parameters for WF model
    :param output: str, output dir
    :param threads: int, number of CPUs
    """
    d_range = np.arange(-1.0, 0.05, 0.05)
    d_range[-1] = 0.0
    h_range = np.arange(0.0, 1.05, 0.05)
    # prepare multiprocessing
    params = [(x, y) for x in h_range for y in d_range]
    iterator = iter(params)
    chunks = takewhile(bool, (list(islice(iterator, 32)) for _ in repeat(None)))
    ready_to_map = [(chunk,  pars, nr_simulations, p0, h_range, d_range, recessive, dominant) for chunk in chunks]
    # multiprocess --> run different parameter combinations in parallel
    pool = mp.Pool(processes=threads)
    error = pool.map(get_error_helper, ready_to_map)
    error = sum(error)
    pool.close()
    pool.join()
    fig, ax = plt.subplots()
    sns.heatmap(error, ax=ax, cmap=sns.diverging_palette(230, 260, s=100, l=40, center='light', as_cmap=True),
                center=0.0, robust=True,
                cbar_kws={'label': r'$P_{fix, emp} - P_{fix, theo}$'}, rasterized=True)
    ax.invert_yaxis()
    ax.invert_xaxis()
    if recessive:
        ax.set_xlabel(r'DMI effects ($\delta_{rd, 2}$)')
    elif dominant:
        ax.set_xlabel(r'DMI effects ($\delta_{dd, 1}$)')
    else:
        ax.set_xlabel(r'DMI effects ($\delta_{sd, 1}$)')
    ax.set_ylabel(r'Heterosis effects ($\eta_1$)')
    ax.set_xticklabels(["{:.2f}".format(l) for l in d_range[np.floor(ax.get_xticks()).astype(int)]], rotation=90)
    ax.set_yticklabels(["{:.2f}".format(l) for l in h_range[np.floor(ax.get_yticks()).astype(int)]], rotation=0)
    if recessive:
        fig.savefig('{}difference_emp_adjusted_theo_fix_recessive_s_{}.pdf'.format(output, pars['s']), dpi=600,
                    bbox_inches='tight')
    elif dominant:
        fig.savefig('{}difference_emp_adjusted_theo_fix_dominant_s_{}.pdf'.format(output, pars['s']), dpi=600,
                    bbox_inches='tight')
    else:
        fig.savefig('{}difference_emp_adjusted_theo_fix_s_{}.pdf'.format(output, pars['s']), dpi=600,
                    bbox_inches='tight')
    plt.close()



def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--population_size', help='Population size, default=10000', type=int, default=10000)
    parser.add_argument('-s', '--selection_coefficients', help='Values for s to try, at most 5.'
                                                               'Enter separated by a space. Default=0 0.005 0.01 0.02 0.03 0.1',
                        nargs='+', action='store', type=float, default=[0, 0.005, 0.01, 0.02, 0.03, 0.1])
    parser.add_argument('-n', '--number_simulations', help='Number of simulations to run, default=100000',
                        default=100000, type=int)
    parser.add_argument('-r', '--recessive', help='DMIs due to recessive-dominant epistasis; default=False',
                        default=False, action='store_true')
    parser.add_argument('--dominant', help='DMIs due to dominant-dominant epistasis; default=False',
                        default=False, action='store_true')
    parser.add_argument('-o', '--output', help='Output directory. Will create a adjusted_pfix.pdf file' +
                                               'in this directory, default=./plots', default='./plots/')
    parser.add_argument('-t', '--threads', help='Number of CPUs, default=16', default=16, type=int)
    parser.add_argument('-e', '--error', default=False, action='store_true', help='Explore error of Eq. 7 '
                                                                                  'on full parameter space')
    args = parser.parse_args()
    # set parameters
    pars = dict()
    pars['h'] = 0.0
    pars['d'] = 0.0
    pars['N'] = 10000
    p0 = 1 / (2 * pars['N'])
    nr_simulations = args.number_simulations
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    if not args.output.endswith('/'):
        output = args.output + '/'
    else:
        output = args.output
    if args.recessive and args.dominant:
        raise AssertionError("Set either -r or -d flag, not both.")
    plot_fixation_probabilities(p0, nr_simulations, args.selection_coefficients, args.recessive, args.dominant, pars,
                                output, args.threads)
    if args.error:
        pars['s'] = args.selection_coefficients[1]
        explore_full_parameter_space(p0, nr_simulations, args.recessive, args.dominant, pars,
                                     output, args.threads)


if __name__ == '__main__':
    main(sys.argv[1:])
