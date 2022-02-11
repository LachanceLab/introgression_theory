#!/usr/bin/env python
import sys
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress
from fixation_probabilities import standard_probability_of_fixation, adjusted_fixation_probability_homogeneous_branching_process,\
    estimate_alpha
from wright_fisher_models import wright_fisher_poisson_decay_hfe, wright_fisher_fixation_prob
from itertools import islice, takewhile, repeat
import multiprocessing as mp
from scipy.stats import fisher_exact


def calculate_fixation_probabilities_helper(args):
    """
    Helper function for multiprocessing calculations of fixation probabilities
    --> explore_parameter_space for the meaning of these parameters
    @param args: list of tuples, (chunks, pars, recessive, dominant, h_range, d_range)
    @return: np.array, (h_range.shape[0], d_range.shape[0]) empirical fixation probabilities
    """
    chunks, pars, recessive, dominant, h_range, d_range = args
    p_fix_array = np.zeros((h_range.shape[0], d_range.shape[0]))
    for chunk in chunks:
        pars['h'], pars['d'] = chunk
        alpha = estimate_alpha(pars['h'], pars['d'], recessive, dominant)
        p_fixation = adjusted_fixation_probability_homogeneous_branching_process(pars['s'], alpha)
        p_fix_array[np.where(h_range == pars['h'])[0],
                    np.where(d_range == pars['d'])[0]] = p_fixation
    return p_fix_array


def explore_parameter_space(p0, recessive, dominant, pars, output, threads):
    """
    Compare fixation probability under our model to fixation probabilities under classical model.
    If both recessive and dominance are False the semidominance-dominance epistasis model is assumed for DMI effects
    :param p0: float, initial allele frequency
    :param recessive: boolean, whether to assume recessive-dominance epistasis model for DMI effects
    :param dominant: boolean, whether to assume dominance-dominance epistasis model for DMI effects
    :param pars: dict, parameters for WF model
    :param output: str, output dir
    :param threads: int, number of processes
    """
    h_range = np.arange(0, 1.01, 0.01)
    d_range = np.arange(-1, 0.01, 0.01)
    d_range[-1] = 0.0
    # prepare multiprocessing
    params = [(x, y) for x in h_range for y in d_range]
    iterator = iter(params)
    chunks = takewhile(bool, (list(islice(iterator, 32)) for _ in repeat(None)))
    ready_to_map = [(chunk,  pars, recessive, dominant, h_range, d_range) for chunk in chunks]
    # multiprocess --> run different parameter combinations in parallel
    pool = mp.Pool(processes=threads)
    emp_fix = pool.map(calculate_fixation_probabilities_helper, ready_to_map)
    # combine results
    emp_fix = sum(emp_fix)
    # close pool
    pool.close()
    pool.join()
    standard_theo_fix = np.full_like(emp_fix, standard_probability_of_fixation(pars['N'], pars['s'], p0))

    fig, ax = plt.subplots()
    sns.heatmap(emp_fix, ax=ax,
                cmap=sns.diverging_palette(12, 260, s=100, l=40, center='light', as_cmap=True),
                center=standard_theo_fix.max(),
                robust=True,
                cbar_kws={'label': 'Fixation probability'},
                rasterized=True)
    ax.invert_yaxis()
    ax.invert_xaxis()
    if recessive:
        ax.set_xlabel(r'DMI effects ($\delta_{rd, 2}$)')
    elif dominant:
        ax.set_xlabel(r'DMI effects ($\delta_{dd, 1}$)')
    else:
        ax.set_xlabel(r'DMI effects ($\delta_{1}$)')
    ax.set_ylabel(r'Heterosis effects ($\eta_1$)')
    print('Max fixation probability: {}'.format(emp_fix.max()))

    value_x = []
    value_y = []
    for i in range(h_range.shape[0]):
        for j in range(d_range.shape[0]):
            if emp_fix[i, j] <= standard_theo_fix.max():
                continue
            else:
                value_x.append(d_range[j])
                value_y.append(h_range[i])
                break
    # linear regression
    value_x = np.abs(value_x)
    reg = linregress(value_x, np.absolute(value_y))
    print('Slope: {:.4f}, y-intercept: {:.4f}, R^2: {:.4f}, pval: {:.4f}'.format(reg.slope, reg.intercept,
                                                                                 reg.rvalue, reg.pvalue))
    ax.plot(h_range[::-1] * h_range.shape[0], ((d_range * reg.slope)[::-1] * -d_range.shape[0]), color='black', ls='--')
    ax.set_xticks(np.arange(0, 105, 5))
    ax.set_yticks(np.arange(0, 105, 5))
    ax.set_xticklabels(["{:.2f}".format(l) for l in np.arange(0, 1.05, 0.05) * -1][::-1], rotation=90)
    ax.set_yticklabels(["{:.2f}".format(l) for l in np.arange(0, 1.05, 0.05)], rotation=0)
    if recessive:
        fig.savefig('{}difference_emp_theo_fix_recessive.pdf'.format(output), dpi=600, bbox_inches='tight')
    elif dominant:
        fig.savefig('{}difference_emp_theo_fix_dominant.pdf'.format(output), dpi=600, bbox_inches='tight')
    else:
        fig.savefig('{}difference_emp_theo_fix.pdf'.format(output), dpi=600, bbox_inches='tight')
    plt.close()


def wright_fisher_helper(args):
    """
    Helper function for multiprocessing simulations of stochastic dilution of donor DNA
    @param args: tuple, initial allele frequency (float), parameters for simulations (dict), number of simulations (int), genome size (int)
    @return: int, number of fixations
    """
    p0, pars, nr_simulations, genome_size = args
    fixations = 0
    for i in range(nr_simulations):
        frequency = wright_fisher_poisson_decay_hfe(p0, pars, genome_size)
        if frequency[-1] == 1.0:
            fixations += 1
    return fixations


def compare_stochastic_vs_deterministic_dilution_of_donor_dna(pars, nr_simulations, threads, genome_size=3200000000):
    """
    Compare the fixation probabilities when assuming a stochastic or deterministic dilution of donor DNA
    @param pars: dict, initial heterosis effects ['h'], initial DMI effects ['d'], effective population size ['N'],
                       selection coefficient ['s']
    @param nr_simulations: int, Number of runs
    @param threads: int, number of threads
    @param genome_size: int, genome size used to determine number of trials in binomial sampling
    """
    p0 = 1 / (2 * pars['N'])
    chunks = np.repeat(64, nr_simulations // 64).tolist()
    if nr_simulations % 64 > 0:
        chunks.append(nr_simulations % 64)
    # stochastic
    ready_to_map = [(p0, pars, c, genome_size) for c in chunks]
    pool = mp.Pool(processes=threads)
    results = pool.map(wright_fisher_helper, ready_to_map)
    fixations = sum(results)
    pool.close()
    pool.join()
    pfix_stochastic = fixations / nr_simulations
    # deterministic
    pfix_deterministic = wright_fisher_fixation_prob(1, pars, False, False, nr_simulations=nr_simulations)
    print("Population size: {}".format(pars['N']))
    print('pfix poisson decay HFEs: {}'.format(pfix_stochastic))
    print('pfix deterministic: {}'.format(pfix_deterministic))
    print("Fisher's exact test: {}".format(fisher_exact([[pfix_stochastic * nr_simulations,
                                                          nr_simulations - (pfix_stochastic * nr_simulations)],
                                                         [pfix_deterministic * nr_simulations,
                                                          nr_simulations - (pfix_deterministic * nr_simulations)]])))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--selection_coefficient', help='Selection coefficient. default=0.01',
                        default=0.01, type=float)
    parser.add_argument('-N', '--population_size', help='Population size. default=10000', default=10000, type=int)
    parser.add_argument('-r', '--recessive', help='DMIs due to recessive-dominant epistasis; default=False',
                        default=False, action='store_true')
    parser.add_argument('-d', '--dominant', help='DMIs due to dominant-dominant epistasis; default=False',
                        default=False, action='store_true')
    parser.add_argument('-o', '--output', help='Output directory. Will create a difference_emp_theo_fix.pdf file' +
                                               'in this directory, default=./plots', default='./plots/')
    parser.add_argument('--stochastic_dilution', help='Simulate stochastic dilution of introgressed DNA',
                        action='store_true')
    parser.add_argument('--genome_size', help='Genome size used to determine number of chunks of 100 Mb'
                                               ' for modeling the dilution of donor with a binomial '
                                               'distribution. Genome size / 100 Mb defines the number of trials. '
                                              '[3200000000]', default=3200000000)
    parser.add_argument('-t', '--threads', help='Number of CPUs, default=16', default=16, type=int)
    args = parser.parse_args()
    # set parameters
    pars = dict()
    pars['s'] = args.selection_coefficient
    pars['N'] = args.population_size
    p0 = 1 / (2 * pars['N'])
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
    explore_parameter_space(p0, recessive, dominant, pars, output, args.threads)

    # simulate stochastic dilution of donor DNA compare to deterministic dilution
    if args.stochastic_dilution:
        heterosis = [0.8, 0]
        dmi_effects = [0, -0.8]
        for h, d in zip(heterosis, dmi_effects):
            print(r'$\eta_1=$' + str(round(h, 3)) + r', $\delta_1$=' + str(round(d, 3)))
            N = [100, 500, 1000, 10000]
            pars['h'] = h
            pars['s'] = 0.01
            pars['d'] = d
            nr_simulations = 1000000
            for n in N:
                pars['N'] = n
                compare_stochastic_vs_deterministic_dilution_of_donor_dna(pars, nr_simulations, args.threads,
                                                                          args.genome_size)


if __name__ == '__main__':
    main(sys.argv[1:])
