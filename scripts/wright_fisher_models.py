#!/usr/bin/python
from fitness import dmi_effect, heterosis_effect, hybrid_fitness, dmi_effect_recessive, dmi_effect_dominant
import numpy as np


def wright_fisher_fixation_prob(n0, pars, recessive=False, dominant=False, nr_simulations=100000):
    """
    Wright Fisher model using fitness function accounting for hybrid effects.
    Specifically for estimating fixation probabilities.
    Workflow taken from https://pycon.org.il/2016/static/sessions/yoav-ram.pdf
    :param n0: int, initials number of allele copies
    :param pars: dict, must hold parameters h, d, s, and N
    :param recessive: boolean, if to assume DMIs due to recessive-dominant interactions
    :param dominant: boolean, if to assume DMIs due to dominant-dominant interactions
    :param nr_simulations: int, number of repetitions
    :return: float fixation probability
    """
    n = np.full(nr_simulations, n0)
    N = pars['N']
    update = np.repeat(True, nr_simulations)
    t = 1
    while update.any():
        if recessive:
            dmi = dmi_effect_recessive(t, pars['d'])
        elif dominant:
            dmi = dmi_effect_dominant(t, pars['d'])
        else:
            dmi = dmi_effect(t, pars['d'])
        het = heterosis_effect(t, pars['h'])
        # calculate hybrid fitness
        w_ab, w_bb = hybrid_fitness(pars['s'], het, dmi)
        # average fitness
        w_bar = ((2 * N - n) / (2 * N)) ** 2 + 2 * ((2 * N - n) / (2 * N)) * n / (2 * N) * w_ab + (n / (2 * N)) ** 2 * w_bb
        # marginal fitness
        w_b = (2 * N - n) / (2 * N) * w_ab + n / (2 * N) * w_bb
        # contribution
        p_fit = (n / (2 * N)) * w_b / w_bar
        p_fit = np.where(p_fit > 1, 1, p_fit)
        # sample
        n[update] = np.random.binomial(2 * N, p_fit[update])
        update = (n > 0) & (n < 2 * N)
        t += 1
    fixation_prob = sum(n == 2 * N) / nr_simulations
    return fixation_prob


def wright_fisher(p0, pars, recessive=False, dominant=False, max_generations=np.inf):
    """
    Wright Fisher model using fitness function accounting for hybrid effects
    :param p0: float, initials frequency of allele
    :param pars: dict, must hold parameters h, d, s, and N
    :param recessive: boolean, if to assume DMIs due to recessive-dominant interactions
    :param dominant: boolean, if to assume DMIs due to dominant-dominant interactions
    :param max_generations: int, maximal number of generations, default=infinity
    :return: list allele frequency over time
    """
    # initialize
    p = p0
    t = 1
    # track fraction of hybrids
    frequency = [0]
    while p != 0.0 and p != 1.0 and t < max_generations:
        if recessive:
            dmi = dmi_effect_recessive(t, pars['d'])
        elif dominant:
            dmi = dmi_effect_dominant(t, pars['d'])
        else:
            dmi = dmi_effect(t, pars['d'])
        het = heterosis_effect(t, pars['h'])
        w_ab, w_bb = hybrid_fitness(pars['s'], het, dmi)
        # average fitness
        w_bar = (1 - p) ** 2 + 2 * (1 - p) * p * w_ab + p ** 2 * w_bb
        # calculate marginal fitness
        w_b = (1 - p) * w_ab + p * w_bb
        # calculate contribution
        p_fit = p * w_b / w_bar
        if p_fit > 1.0:
            p_fit = 1.0
        # sample
        n1 = np.random.binomial(2 * pars['N'], p_fit)
        # calculate updated fraction of hybrids
        p = n1 / (2 * pars['N'])
        frequency.append(p)
        t += 1
    return frequency


def standard_wright_fisher(p0, s, N):
    """
    Standard Wright Fisher assuming dominance
    :param p0: float, initial allele frequency
    :param s: float, selection coefficient
    :param N: int, population size
    :return: list, fraction of allele over time
    """
    # initialize
    p = p0
    w_ab = (1 + s)
    w_bb = (1 + 2 * s)
    # track fraction of introgressed alleles
    frequency = [0]
    while p != 0.0 and p != 1.0:
        # average fitness
        w_bar = (1 - p) ** 2 + 2 * (1 - p) * p * w_ab + p ** 2 * w_bb
        # calculate marginal fitness
        w_b = (1 - p) * w_ab + p * w_bb
        # contribution
        p_fit = p * w_b / w_bar
        if p_fit > 1.0:
            p_fit = 1.0
        # sample
        n1 = np.random.binomial(2 * N, p_fit)
        # calculate updated fraction of introgressed alleles
        p = n1 / (2 * N)
        frequency.append(p)

    return frequency
