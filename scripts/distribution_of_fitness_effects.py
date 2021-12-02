#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import os
from wright_fisher_models import wright_fisher, wright_fisher_fixation_prob
from scipy.stats import norm, expon
from itertools import islice, takewhile, repeat
import multiprocessing as mp


def dfe_simulations_helper(args):
    """
    Helper function to parallelize approximation of DFE conditioned on fixation
    @param args: tuple, (s_vals, pars, p0, recessive, dominant) for meaning of parameters see dfe_normal_simulations
    @return: np.array, selection coefficients that reached fixations
    """
    s_vals, pars, p0, recessive, dominant = args
    fixations = []
    for s in s_vals:
        pars['s'] = s
        frequency = wright_fisher(p0, pars, recessive, dominant)
        if frequency[-1] == 1.0:
            fixations.append(s)
    return np.array(fixations)


def dfe_normal_simulations(N, mean_normal, std_normal, recessive, dominant, threads, nr_simulations, output):
    """
    Infer distribution of fitness effects conditioned on fixation from simulations. Orignal DFE is a normal
    distribution.
    @param N: int, population size
    @param mean_normal: float, mean of normal distribution
    @param std_normal: float, standard deviation of normal distribution
    @param recessive: boolean, whether to assume recessive-dominance epistasis model for DMI effects
    @param dominant: boolean, whether to assume dominance-dominance epistasis model for DMI effects
    @param threads: int, number of CPUs
    @param nr_simulations: int, number simulations to run
    @param output: str, directory where to save figures
    """
    normal = norm(loc=mean_normal, scale=std_normal)
    s_vals = normal.rvs(nr_simulations)
    heterosis = [0.0, 0.8, 0.0, 0.8]
    dmis = [0.0, 0.0, -0.8, -0.8]
    colors = ["blue", "deepskyblue", "lightgrey", "grey",]
    fixed_s_values = dfe_simulations(N, s_vals, recessive, dominant, threads)
    fig, ax = plt.subplots()

    # inset with original DFE
    inset = ax.inset_axes([0.59, 0.59, 0.4, 0.4])
    inset.text(0.05, 0.9, 'original DFE', fontsize=8, transform=inset.transAxes)
    inset.hist(s_vals, histtype='step', color='black', bins=250, weights=np.full(s_vals.shape[0], 1 / s_vals.shape[0]))
    inset.axvline(0.0, ls='--', color='gray', lw=1)
    inset.set_ylabel(r'P(f(s))', fontsize=8)
    inset.set_xlabel(r's', fontsize=8, labelpad=2)
    inset.tick_params(axis='x', labelsize=6)
    inset.tick_params(axis='y', labelsize=6)

    for i, (fixed_s, h, d, c) in enumerate(zip(fixed_s_values, heterosis, dmis, colors)):
        if fixed_s.shape[0] == 0:
            continue
        if recessive:
            hist, bins, _ = ax.hist(fixed_s, bins=250, color=c, histtype='step',
                                    weights=np.full(fixed_s.shape[0], 1 / fixed_s.shape[0]),
                                    label=r'$\eta_1=$' + str(round(h, 3)) + r', $\delta_{rd, 2}=$' + str(round(d, 3)))
        elif dominant:
            hist, bins, _ = ax.hist(fixed_s, bins=250, color=c, histtype='step',
                                    weights=np.full(fixed_s.shape[0], 1 / fixed_s.shape[0]),
                                    label=r'$\eta_1=$' + str(round(h, 3)) + r', $\delta_{dd, 1}=$' + str(round(d, 3)))
        else:
            hist, bins, _ = ax.hist(fixed_s, bins=250, color=c, histtype='step',
                                    weights=np.full(fixed_s.shape[0], 1 / fixed_s.shape[0]),
                                    label=r'$\eta_1=$' + str(round(h, 3)) + r', $\delta_{sd, 1}=$' + str(round(d, 3)))
        ax.text(0.6, 0.45 - i * 0.05, r'$\overline{s}=$' + str(round(fixed_s.mean(), 4)), color=c, transform=ax.transAxes,
                fontsize=10)

    ax.set_xlabel('selection coefficient (s)', fontsize=10)
    ax.set_ylabel(r'P(f(s) | fixation)', fontsize=10)
    ax.legend(bbox_to_anchor=(0.5, -.14), loc='upper center', ncol=2)
    if recessive:
        fig.savefig('{}dfe_normal_simulations_recessive.pdf'.format(output), bbox_inches='tight', dpi=600)
    elif dominant:
        fig.savefig('{}dfe_normal_simulations_dominant.pdf'.format(output), bbox_inches='tight', dpi=600)
    else:
        fig.savefig('{}dfe_normal_simulations.pdf'.format(output), bbox_inches='tight', dpi=600)
    plt.close()


def dfe_joint_normal_simulations_helper(args):
    """
    Helper function to parallelize approximation of joint distribution of DFEs and fixation probabilities
    @param args: tuple, (iterations, pars, p0, recessive, dominant) for meaning of parameters see
                        dfe_joint_normal_simulations
    @return: int, fixation probability
    """
    iterations, pars, p0, recessive, dominant = args
    p_fixation = wright_fisher_fixation_prob(int(p0 * 2 * pars['N']), pars, recessive, dominant, iterations)

    return p_fixation


def dfe_joint_normal_simulations(N, mean_normal, std_normal, recessive, dominant, threads, nr_simulations, output):
    """
    Approximate joint distribution of original DFE - a normal distribution with mean mean_normal and standard deviation
    std_normal, and fixation probability
    @param N: int, population size
    @param mean_normal: float, mean of normal distribution
    @param std_normal: float, standard deviation of normal distribution
    @param recessive: boolean, whether to assume recessive-dominance epistasis model for DMI effects
    @param dominant: boolean, whether to assume dominance-dominance epistasis model for DMI effects
    @param threads: int, number of CPUs
    @param nr_simulations: int, number simulations to run
    @param output: str, directory where to save figures
    """
    normal = norm(loc=mean_normal, scale=std_normal)
    svals = np.arange(-0.05, 0.2, 0.005)
    heterosis = [0.0, 0.8, 0.0, 0.8]
    dmis = [0.0, 0.0, -0.8, -0.8]
    colors = ["blue", "deepskyblue", "lightgrey", "grey",]
    pars = dict()
    pars['N'] = N
    p0 = 1 / (2 * N)

    fig, ax = plt.subplots()
    for i, (h, d, c) in enumerate(zip(heterosis, dmis, colors)):
        pars['h'] = h
        pars['d'] = d
        p_fix_s = []
        for s in svals:
            pars['s'] = s
            iterations_per_batch = [256] * (nr_simulations // 256)
            iterations_per_batch.append(nr_simulations % 256)
            ready_to_map = [(iterations, pars, p0, recessive, dominant) for iterations in iterations_per_batch]
            pool = mp.Pool(processes=threads)
            p_fixations = pool.map(dfe_joint_normal_simulations_helper, ready_to_map)
            p_fix_s.append(np.array(p_fixations).mean() * normal.pdf(s))
            pool.close()
            pool.join()
        if recessive:
            ax.plot(svals, p_fix_s, color=c,
                    label=r'$\eta_1=$' + str(round(h, 3)) + r', $\delta_{rd, 2}=$' + str(round(d, 3)))
        elif dominant:
            ax.plot(svals, p_fix_s, color=c,
                    label=r'$\eta_1=$' + str(round(h, 3)) + r', $\delta_{dd, 1}=$' + str(round(d, 3)))
        else:
            ax.plot(svals, p_fix_s, color=c,
                    label=r'$\eta_1=$' + str(round(h, 3)) + r', $\delta_{sd, 1}=$' + str(round(d, 3)))
    ax.set_xlabel('selection coefficient (s)')
    ax.set_ylabel(r'P(f(s), u(s))')
    ax.legend(bbox_to_anchor=(0.5, -.14), loc='upper center', ncol=2)
    if recessive:
        fig.savefig('{}joint_dfe_normal_simulations_recessive.pdf'.format(output), bbox_inches='tight', dpi=600)
    elif dominant:
        fig.savefig('{}joint_dfe_normal_simulations_dominant.pdf'.format(output), bbox_inches='tight', dpi=600)
    else:
        fig.savefig('{}joint_dfe_normal_simulations.pdf'.format(output), bbox_inches='tight', dpi=600)
    plt.close()


def dfe_simulations(N, s_vals, recessive, dominant, threads):
    """
    Infer distribution of fitness effects conditioned on fixation from simulations.
    @param N: int, population size
    @param s_vals: np.array, random variables sampled from corresponding distribution for DFEs
    @param recessive: boolean, whether to assume recessive-dominance epistasis model for DMI effects
    @param dominant: boolean, whether to assume dominance-dominance epistasis model for DMI effects
    @param threads: int, number of CPUs
    @return: np.array, selection coefficents that reached fixation
    """
    heterosis = [0.0, 0.8, 0.0, 0.8]
    dmis = [0.0, 0.0, -0.8, -0.8]
    colors = ['blue', 'green', 'red', 'orange']
    pars = dict()
    pars['N'] = N
    p0 = 1 / (2 * N)
    fixed_s_values = []
    for i, (h, d, c) in enumerate(zip(heterosis, dmis, colors)):
        pars['h'] = h
        pars['d'] = d
        iterator = iter(s_vals)
        chunks = takewhile(bool, (list(islice(iterator, 256)) for _ in repeat(None)))
        ready_to_map = [(s_vals, pars, p0, recessive, dominant) for s_vals in chunks]
        pool = mp.Pool(processes=threads)
        fixed_s = pool.map(dfe_simulations_helper, ready_to_map)
        fixed_s = np.concatenate(fixed_s)
        fixed_s_values.append(fixed_s)
        pool.close()
        pool.join()
    return fixed_s_values


def dfe_benefical_simulations_plotting_helper(fixed_s_values, recessive, dominant,
                                              heterosis, dmis, colors, N, ax):
    """
    Helper function to plot DFEs relative to DFE under the classical model
    @param fixed_s_values: list, list of np.arrays of selection coefficient that reached fixation under
                                 the different conditons
    @param recessive: boolean, whether to assume recessive-dominance epistasis model for DMI effects
    @param dominant: boolean, whether to assume dominance-dominance epistasis model for DMI effects
    @param heterosis: list, heterosis parameters
    @param dmis: list, dmi parameters
    @param colors: list, colors for different parameter combinations
    @param N: int, population size
    @param ax: Axes object, ax to plot on
    @return: ax
    """
    hist_classical, bins_classical = np.histogram(fixed_s_values[0], bins=250,
                                                  range=(min([min(x) for x in fixed_s_values if x.shape[0] > 0]),
                                                         max([max(x) for x in fixed_s_values if x.shape[0] > 0])),
                                                  weights=np.full(fixed_s_values[0].shape[0],
                                                                  1 / fixed_s_values[0].shape[0]))

    for i, (fixed_s, h, d, c) in enumerate(zip(fixed_s_values, heterosis, dmis, colors)):
        if fixed_s.shape[0] == 0:
            continue
        if i == 0:
            if recessive:
                ax.plot(bins_classical[:-1] * N, hist_classical, color=c,
                        label=r'$\eta_1=$' + str(round(h, 3)) + r', $\delta_{rd, 2}=$' + str(round(d, 3)))
            elif dominant:
                ax.plot(bins_classical[:-1] * N, hist_classical, color=c,
                        label=r'$\eta_1=$' + str(round(h, 3)) + r', $\delta_{dd, 1}=$' + str(round(d, 3)))
            else:
                ax.plot(bins_classical[:-1] * N, hist_classical, color=c,
                        label=r'$\eta_1=$' + str(round(h, 3)) + r', $\delta_{sd, 1}=$' + str(round(d, 3)))
        else:
            hist, bins = np.histogram(fixed_s, bins=bins_classical,
                                      weights=np.full(fixed_s.shape[0], 1 / fixed_s.shape[0]))
            # hist += 1
            if recessive:
                ax.plot(bins_classical[:-1] * N, hist, color=c,
                        label=r'$\eta_1=$' + str(round(h, 3)) + r', $\delta_{rd, 2}=$' + str(round(d, 3)))
            elif dominant:
                ax.plot(bins_classical[:-1] * N, hist, color=c,
                        label=r'$\eta_1=$' + str(round(h, 3)) + r', $\delta_{dd, 1}=$' + str(round(d, 3)))
            else:
                ax.plot(bins_classical[:-1] * N, hist, color=c,
                        label=r'$\eta_1=$' + str(round(h, 3)) + r', $\delta_{sd, 1}=$' + str(round(d, 3)))
        ax.text(0.7, 0.45 - i * 0.05, r'$N_e\overline{s}=$' + str(round(fixed_s.mean() * N, 2)), color=c, transform=ax.transAxes,
                fontsize=10)
    return ax


def dfe_beneficial_simulations(population_size, mean_exp, recessive, dominant,
                               threads, nr_simulations_dfe, output, ):
    """
    Plot DFEs of beneficial alleles after selection relative to DFE after selection under the classical.
    The original DFE of beneficial mutations is model by an exponential distribution.
    @param population_size: int, population size
    @param mean_exp: float, mean of exponential
    @param recessive: boolean, whether to assume recessive-dominance epistasis model for DMI effects
    @param dominant: boolean, whether to assume dominance-dominance epistasis model for DMI effects
    @param threads: int, number of CPUs
    @param nr_simulations_dfe: int, number of selection coefficients to draw
    @param output: str, path to directory where to save plot
    """
    heterosis = [0.0, 0.8, 0.0, 0.8]
    dmis = [0.0, 0.0, -0.8, -0.8]
    colors = ["blue", "deepskyblue", "lightgrey", "grey",]
    fig, ax = plt.subplots()
    exponential = expon(scale=mean_exp)
    s_vals = exponential.rvs(nr_simulations_dfe)

    fixed_s_values = dfe_simulations(population_size, s_vals, recessive, dominant, threads)
    ax = dfe_benefical_simulations_plotting_helper(fixed_s_values, recessive, dominant, heterosis,
                                                   dmis, colors, population_size, ax)
    # inset with original DFE
    inset = ax.inset_axes([0.59, 0.59, 0.4, 0.4])
    inset.text(0.1, 0.9, 'original DFE', fontsize=8, transform=inset.transAxes)
    inset.hist(s_vals, histtype='step', color='black', bins=250, weights=np.full(s_vals.shape[0], 1 / s_vals.shape[0]))
    inset.set_ylabel(r'P(f(s))', fontsize=8)
    inset.set_xlabel(r's', fontsize=8, labelpad=2)
    inset.tick_params(axis='x', labelsize=6)
    inset.tick_params(axis='y', labelsize=6)

    ax.set_xlabel(r'$N_es$')
    ax.set_xlim([0, 600])
    if recessive:
        ax.set_ylabel(r'P(f(s) | fixation)', fontsize=10)
    elif dominant:
        ax.set_ylabel(r'P(f(s) | fixation)', fontsize=10)
    else:
        ax.set_ylabel(r'P(f(s) | fixation)', fontsize=10)
    ax.legend(bbox_to_anchor=(0.5, -.14), loc='upper center', ncol=2)
    if recessive:
        fig.savefig('{}dfe_exponential_simulations_recessive.pdf'.format(output), bbox_inches='tight', dpi=600)
    elif dominant:
        fig.savefig('{}dfe_exponential_simulations_dominant.pdf'.format(output), bbox_inches='tight', dpi=600)
    else:
        fig.savefig('{}dfe_exponential_simulations.pdf'.format(output), bbox_inches='tight', dpi=600)
    plt.close()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--population_size', help='Effective population size, default=10000', default=10000,
                        type=int)
    parser.add_argument('-m1', '--mean_exp', help='Mean of exponential distribution, default=0.0085', default=0.0085,
                        type=float)
    parser.add_argument('-m2', '--mean_normal', help='Mean of normal distribution, default=-0.001', default=-0.001,
                        type=float)
    parser.add_argument('-s1', '--std_normal', help='Standard deviation of normal distribution, default=0.05',
                        default=0.05, type=float)
    parser.add_argument('-r', '--recessive', help='DMIs due to recessive-dominant epistasis; default=False',
                        default=False, action='store_true')
    parser.add_argument('-d', '--dominant', help='DMIs due to dominant-dominant epistasis; default=False',
                        default=False, action='store_true')
    parser.add_argument('-o', '--output', help='Output directory. Will create a dfe_exponential.pdf file' +
                                               'in this directory, default=./plots', default='./plots/')
    parser.add_argument('-t', '--threads', help='Number of CPUs, default=16', default=16, type=int)
    parser.add_argument('-n1', '--nr_simulations_dfe', help='Number of selection values to draw to infer'
                                                            'conditional DFE from simulations, default=10000000',
                        type=int, default=10000000)
    parser.add_argument('-n2', '--nr_simulations_joint_dfe', help='Number of simulations to run to infer fixation '
                                                                  'probability, default=100000',
                        type=int, default=100000)
    parser.add_argument('-j', '--joint', default=False, action='store_true', help='Plot joint distribution of DFE and '
                                                                                  'fixation probabilities')
    parser.add_argument('-b', '--beneficial', default=False, action='store_true',
                        help='Plot DFE after selection of beneficial mutation. Original DFE is modeled with an '
                             'exponential distribution. Use -m1 to set mean of that distribution')
    args = parser.parse_args()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    if not args.output.endswith('/'):
        output = args.output + '/'
    else:
        output = args.output
    if args.recessive and args.dominant:
        raise AssertionError("Set either -r or -d flag, not both.")
    dfe_normal_simulations(args.population_size, args.mean_normal, args.std_normal, args.recessive, args.dominant,
                           args.threads, args.nr_simulations_dfe, output)
    if args.joint:
        dfe_joint_normal_simulations(args.population_size, args.mean_normal, args.std_normal, args.recessive, args.dominant,
                                     args.threads, args.nr_simulations_joint_dfe, output)
    if args.beneficial:
        dfe_beneficial_simulations(args.population_size, args.mean_exp,
                                   args.recessive, args.dominant, args.threads, args.nr_simulations_dfe,
                                   output)


if __name__ == '__main__':
    main(sys.argv[1:])
