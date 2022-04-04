#!/usr/bin/env python
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
from fixation_probabilities import adjusted_probability_of_fixation_diffusion_approximation, estimate_alpha, \
    adjusted_fixation_probability_heterogeneous_branching_process
from wright_fisher_models import wright_fisher, wright_fisher_fixation_prob
from itertools import islice, takewhile, repeat
import multiprocessing as mp


def assess_fixation_probability_helper(args):
    """
    Helper function for multiprocessing the assessment of the fixation probability formula.
    Runs the actual simulations over the parameter space (heterosis, DMIs)
    @param args: list of tuples, Input variables. See plot_fixation_probabilities for the meaning of the parameters
    @return: (np.array, np.array, np.array), empirical fixation probability, diffusion approximation,
                                            and branching process (val_range.shape[0])
    """
    vals, pars, p0, recessive, dominant, nr_simulations, val_range, heterosis, dmis = args
    emp_fix = np.zeros_like(val_range)
    theo_fix_da = np.zeros_like(val_range)
    theo_fix_bp = np.zeros_like(val_range)
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
        theo_p_fixation_da = adjusted_probability_of_fixation_diffusion_approximation(pars['N'], pars['s'], p0, alpha)
        theo_p_fixation_bp = adjusted_fixation_probability_heterogeneous_branching_process(pars, recessive=recessive,
                                                                                           dominant=dominant)
        theo_fix_da[np.where(val_range == val)[0]] = theo_p_fixation_da
        theo_fix_bp[np.where(val_range == val)[0]] = theo_p_fixation_bp

    return emp_fix, theo_fix_da, theo_fix_bp


def plot_fixation_probabilities(p0, nr_simulations, values_s, recessive, dominant, pars, output, threads):
    """
    Plot the empirical and theoretical fixation probability for different strengths of heterosis and DMI effects.
    If both recessive and dominant are false the semidominance-dominance epistasis model is assumed for DMI effects.
    :param p0: float, initial allele frequency
    :param nr_simulations: int, number of simulations to run
    :param values_s: list, values for s to use
    :param recessive: boolean, whether to assume recessive-dominance epistasis model for DMI effects
    :param dominant: boolean, whether to assume dominance-dominance epistasis model for DMI effects
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
        theo_fix_da = sum([result[1] for result in results])
        theo_fix_bp = sum([result[2] for result in results])
        pool.close()
        pool.join()
        ax[0].plot(h_range, theo_fix_da, ls=':', color=c)
        if s > 0:
            ax[0].plot(h_range, theo_fix_bp, ls='--', color=c)
        ax[0].scatter(h_range[::4], emp_fix[::4], marker=marker, label='s={:.3f}'.format(s),
                      color=c)
    ax[0].text(-0.1, 1.05, 'A', transform=ax[0].transAxes, fontsize=12, weight='bold')
    ax[0].set_xlabel(r'Heterosis effects ($\eta_1$)')

    ax[0].set_ylabel("Probability of fixation")
    # add dummy to create a second row
    handles, labels = ax[0].get_legend_handles_labels()
    handles.insert(1, Line2D([], [], marker='o', markerfacecolor='none', markeredgecolor='none', ls=''))
    labels.insert(1, '')
    ax[0].legend(handles, labels, loc='upper center', bbox_to_anchor=(1.05, -.14), ncol=len(values_s), labelspacing=2,
                 columnspacing=2, handletextpad=0.5)

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
        theo_fix_da = sum([result[1] for result in results])
        theo_fix_bp = sum([result[2] for result in results])
        pool.close()
        pool.join()
        ax[1].plot(d_range, theo_fix_da, ls=':', color=c)
        ax[1].plot(d_range, theo_fix_bp, ls='--', color=c)
        ax[1].scatter(d_range[::4], emp_fix[::4], marker=marker, color=c)
    ax[1].text(-0.1, 1.05, 'B', transform=ax[1].transAxes, fontsize=12, weight='bold')
    if recessive:
        ax[1].set_xlabel(r'DMI effects ($\delta_{rd, 2}$)')
    elif dominant:
        ax[1].set_xlabel(r'DMI effects ($\delta_{dd, 1}$)')
    else:
        ax[1].set_xlabel(r'DMI effects ($\delta_{1}$)')

    handles = [Line2D([0], [0], ls='--', color='black'), Line2D([0], [0], ls=':', color='black'),
               Line2D([0], [0], ls='', color='black', marker='o')]
    labels = [r'Branching Process', 'Diffusion approximation', 'WF simulations']
    ax[1].legend(handles, labels, loc='upper center', bbox_to_anchor=(-0.10, -.21), ncol=4, frameon=False,
                 columnspacing=1, handletextpad=0.5)
    if recessive:
        fig.savefig('{}adjusted_pfix_recessive.pdf'.format(output), dpi=600, bbox_inches='tight')
    elif dominant:
        fig.savefig('{}adjusted_pfix_dominant.pdf'.format(output), dpi=600, bbox_inches='tight')
    else:
        fig.savefig('{}adjusted_pfix.pdf'.format(output), dpi=600, bbox_inches='tight')
    plt.close()


def fixation_probabilities_neutral_alleles_helper(args):
    p0, pars, nr_simulations, recessive, dominant = args
    fixations = 0
    for i in range(nr_simulations):
        frequency = wright_fisher(p0, pars, recessive, dominant)
        if frequency[-1] == 1.0:
            fixations += 1
    return fixations


def fixation_probabilities_neutral_alleles(N, nr_simulations, recessive, dominant, output, threads):
    """
    Assess adjusted fixation probability of neutral alleles
    @param N: int, population size
    @param nr_simulations: int, number of WF simulations to run
    @param recessive: boolean, whether to assume recessive-dominance epistasis model for DMI effects
    @param dominant: boolean, whether to assume dominance-dominance epistasis model for DMI effects
    @param output: str, output directory
    @param threads: int, number of threads

    """
    pars = dict()
    pars['N'] = N
    pars['s'] = 0.0
    p0 = 1 / (2 * pars['N'])
    heterosis_effects = [0.8, 0.4, 0, 0.8,  0, 0]
    dmi_effects = [0, 0, 0, -0.8, -0.4, -0.8]
    colors = ["blue", "cornflowerblue", "deepskyblue", "darkorange", "orangered", 'red']
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], ls='--', color='black')
    observed_pfix = []
    alpha_2N_exp = []
    for i, (h, d, c) in enumerate(zip(heterosis_effects, dmi_effects, colors)):
        pars['h'] = h
        pars['d'] = d
        chunks = np.repeat(64, nr_simulations // 64).tolist()
        if nr_simulations % 64 > 0:
            chunks.append(nr_simulations % 64)
        ready_to_map = [(p0, pars, c, recessive, dominant) for c in chunks]
        pool = mp.Pool(processes=threads)
        results = pool.map(fixation_probabilities_neutral_alleles_helper, ready_to_map)
        fixations = sum(results)
        observed_pfix.append(fixations / nr_simulations)
        alpha_2N_exp.append(estimate_alpha(pars['h'], pars['d'], recessive, dominant) / (2 * pars['N']))
        pool.close()
        pool.join()
        ax.scatter(estimate_alpha(pars['h'], pars['d'], recessive, dominant) / (2 * pars['N']),
                   fixations / nr_simulations, color=c,
                   label=r'$\eta_1=$' + str(round(h, 3)) + r', $\delta_{1}=$' + str(round(d, 3)))
    ax.set_xlabel(r'$p_{fix}=\alpha/2N$')
    ax.set_ylabel(r'$p_{fix}$ from WF simulations')
    ax.set_xlim([0, max([max(observed_pfix), max(alpha_2N_exp)]) + 0.00005])
    ax.set_ylim([0, max([max(observed_pfix), max(alpha_2N_exp)]) + 0.00005])
    ax.legend(bbox_to_anchor=(0.5, -.15), ncol=2, loc='upper center')
    rmse = (sum((np.array(alpha_2N_exp) - np.array(observed_pfix)) ** 2) / len(alpha_2N_exp)) ** 1/2
    print("Fixation probabilities neutral alleles")
    print("RMSE: {}".format(rmse))
    if recessive:
        fig.savefig(f'{output}fixation_probability_neutral_recessive.pdf', bbox_inches='tight', dpi=600)
    elif dominant:
        fig.savefig(f'{output}fixation_probability_neutral_dominant.pdf', bbox_inches='tight', dpi=600)
    else:
        fig.savefig(f'{output}fixation_probability_neutral.pdf', bbox_inches='tight', dpi=600)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--population_size', help='Population size, default=10000', type=int, default=10000)
    parser.add_argument('-s', '--selection_coefficients', help='Values for s to try, at most 5.'
                                                               'Enter separated by a space. Default=0 0.005 0.01 0.02 0.03 0.1',
                        nargs='+', action='store', type=float, default=[0.005, 0.01, 0.02, 0.03, 0.1])
    parser.add_argument('-n', '--number_simulations', help='Number of simulations to run, default=100000',
                        default=100000, type=int)
    parser.add_argument('-r', '--recessive', help='DMIs due to recessive-dominant epistasis; default=False',
                        default=False, action='store_true')
    parser.add_argument('--dominant', help='DMIs due to dominant-dominant epistasis; default=False',
                        default=False, action='store_true')
    parser.add_argument('-o', '--output', help='Output directory. Will create a adjusted_pfix.pdf file' +
                                               'in this directory, default=./plots', default='./plots/')
    parser.add_argument('-t', '--threads', help='Number of CPUs, default=16', default=16, type=int)
    parser.add_argument('--neutral', help="Assess adjusted fixation probabilities of neutral alleles",
                        action='store_true', default=False)
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
    if args.neutral:
        fixation_probabilities_neutral_alleles(pars['N'], nr_simulations, args.recessive, args.dominant, output,
                                               args.threads)


if __name__ == '__main__':
    main(sys.argv[1:])
