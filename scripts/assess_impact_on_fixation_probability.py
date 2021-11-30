#!/usr/bin/python
import sys
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress
from fixation_probabilities import standard_probability_of_fixation
from wright_fisher_models import wright_fisher, wright_fisher_fixation_prob
from itertools import islice, takewhile, repeat
import multiprocessing as mp


def run_wright_fisher_helper(args):
    """
    Helper function for multiprocessing simulations --> explore_parameter_space for the meaning of these parameters
    @param args: list of tuples, (chunks, pars, p0, recessive, dominant, nr_simulations, h_range, d_range)
    @return: np.array, (h_range.shape[0], d_range.shape[0]) empirical fixation probabilities
    """
    chunks, pars, p0, recessive, dominant, nr_simulations, h_range, d_range = args
    emp_fix = np.zeros((h_range.shape[0], d_range.shape[0]))
    for chunk in chunks:
        pars['h'], pars['d'] = chunk
        emp_p_fixation = wright_fisher_fixation_prob(int(p0 * 2 * pars['N']), pars, recessive, dominant, nr_simulations)
        emp_fix[np.where(h_range == pars['h'])[0],
                np.where(d_range == pars['d'])[0]] = emp_p_fixation
    return emp_fix


def explore_parameter_space(p0, nr_simulations, recessive, dominant, pars, output, threads):
    """
    Compare fixation probability under our model to fixation probabilities under classical model.
    If both recessive and dominance are False the semidominance-dominance epistasis model is assumed for DMI effects
    :param p0: float, initial allele frequency
    :param nr_simulations: int, number of simulations to run
    :param recessive: boolean, whether to assume recessive-dominance epistasis model for DMI effects
    :param dominant: boolean, whether to assume dominance-dominance epistasis model for DMI effects
    :param pars: dict, parameters for WF model
    :param output: str, output dir
    :param threads: int, number of processes
    """
    h_range = np.arange(0, 1.025, 0.025)
    d_range = np.arange(-1, 0.025, 0.025)
    d_range[-1] = 0.0
    # prepare multiprocessing
    params = [(x, y) for x in h_range for y in d_range]
    iterator = iter(params)
    chunks = takewhile(bool, (list(islice(iterator, 32)) for _ in repeat(None)))
    ready_to_map = [(chunk,  pars, p0, recessive, dominant, nr_simulations, h_range, d_range) for chunk in chunks]
    # multiprocess --> run different parameter combinations in parallel
    pool = mp.Pool(processes=threads)
    emp_fix = pool.map(run_wright_fisher_helper, ready_to_map)
    # combine results
    emp_fix = sum(emp_fix)
    # close pool
    pool.close()
    pool.join()
    standard_theo_fix = np.full_like(emp_fix, standard_probability_of_fixation(pars['N'], pars['s'], p0))

    fig, ax = plt.subplots()
    sns.heatmap(emp_fix, ax=ax,
                cmap=sns.diverging_palette(230, 260, s=100, l=40, center='light', as_cmap=True),
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
        ax.set_xlabel(r'DMI effects ($\delta_{sd, 1}$)')
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
    ax.set_xticklabels(["{:.2f}".format(l) for l in d_range[np.floor(ax.get_xticks()).astype(int)]], rotation=90)
    ax.set_yticklabels(["{:.2f}".format(l) for l in h_range[np.floor(ax.get_yticks()).astype(int)]], rotation=0)
    if recessive:
        fig.savefig('{}difference_emp_theo_fix_recessive.pdf'.format(output), dpi=600, bbox_inches='tight')
    elif dominant:
        fig.savefig('{}difference_emp_theo_fix_dominant.pdf'.format(output), dpi=600, bbox_inches='tight')
    else:
        fig.savefig('{}difference_emp_theo_fix.pdf'.format(output), dpi=600, bbox_inches='tight')
    plt.close()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--selection_coefficient', help='Selection coefficient. default=0.01',
                        default=0.01, type=float)
    parser.add_argument('-N', '--population_size', help='Population size. default=10000', default=10000, type=int)
    parser.add_argument('-n', '--number_simulations', help='Number of simulations to run. default=100000',
                        default=100000, type=int)
    parser.add_argument('-r', '--recessive', help='DMIs due to recessive-dominant epistasis; default=False',
                        default=False, action='store_true')
    parser.add_argument('-d', '--dominant', help='DMIs due to dominant-dominant epistasis; default=False',
                        default=False, action='store_true')
    parser.add_argument('-o', '--output', help='Output directory. Will create a difference_emp_theo_fix.pdf file' +
                                               'in this directory, default=./plots', default='./plots/')
    parser.add_argument('-t', '--threads', help='Number of CPUs, default=16', default=16, type=int)
    args = parser.parse_args()
    # set parameters
    pars = dict()
    pars['s'] = args.selection_coefficient
    pars['N'] = args.population_size
    p0 = 1 / (2 * pars['N'])
    nr_simulations = args.number_simulations
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
    explore_parameter_space(p0, nr_simulations, recessive, dominant, pars, output, args.threads)


if __name__ == '__main__':
    main(sys.argv[1:])
