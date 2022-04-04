#!/usr/bin/env python
import numpy as np
from fitness import heterosis_effect, dmi_effect, dmi_effect_recessive, dmi_effect_dominant, hybrid_fitness
from scipy.stats import poisson
from scipy.optimize import fsolve


def estimate_alpha(h, d, recessive, dominant):
    """
    Compute compounded hybrid effects
    :param h: float, initial strength of heterosis
    :param d: float, initial strength of DMI effects
    :param recessive: boolean, if recessive-dominance epistasis model should be assumed for DMI effects
    :param dominant: boolean, if dominance-dominance epistasis model should be assumed for DMI effects
    :return: float, compounded hybrid effects
    """
    t = np.arange(0, 20)
    heterosis = heterosis_effect(t, h)
    if recessive:
        dmi = dmi_effect_recessive(t, d)
    elif dominant:
        dmi = dmi_effect_dominant(t, d)
    else:
        dmi = dmi_effect(t, d)
    return np.prod(1 + heterosis[1:] + dmi[1:])


def adjusted_probability_of_fixation_diffusion_approximation(N, s, p, alpha):
    """
    Compute fixation probability under consideration of hybrid effects
    :param N: int, population size
    :param s: float, selection coefficient
    :param p: float, initial frequency of allele
    :param alpha: float, compounded hybrid effects
    :return: float, probability of fixation
    """

    numerator = 1 - np.exp(-4 * N * s * alpha * p)
    denominator = 1 - np.exp(-4 * N * s)
    if denominator == 0 or (np.abs(s) <= 1 / N or np.isclose(s, 0.0)):
        return alpha / (2 * N)
    elif denominator == np.inf:
        return 0.0
    else:
        return numerator / denominator


def standard_probability_of_fixation(N, s, p):
    """
    Compute standard probability of fixation according to Kimura
    :param N: int, population size
    :param s: float, selection coefficient
    :param p: float, initial frequency of allele
    :return: float, probability of fixation
    """
    numerator = 1 - np.exp(-4 * N * s * p)
    denominator = 1 - np.exp(-4 * N * s)
    if denominator == 0 or (np.abs(s) <= 1 / N or np.isclose(s, 0.0)):
        return 1 / (2 * N)
    elif denominator == np.inf:
        return 0.0
    else:
        return numerator / denominator


def heterogeneous_branching_process_helper(pars, t=1, n=800, recessive=False, dominant=False):
    """
    Recursion equation for heterogeneous branching process (Equation S4)
    @param pars: dict, initial heterosis effects ['h'], initial DMI effects ['d'], and selection coefficient ['s']
    @param t: int, current generation >= 1
    @param n: int, final time
    @param recessive: boolean, if recessive-dominance epistasis model should be assumed for DMI effects
    @param dominant: boolean, if dominance-dominance epistasis model should be assumed for DMI effects
    @return: float, extinction probability
    """
    if t == n:
        het = heterosis_effect(t, pars['h'])
        if recessive:
            dmi = dmi_effect_recessive(t, pars['d'])
        elif dominant:
            dmi = dmi_effect_dominant(t, pars['d'])
        else:
            dmi = dmi_effect(t, pars['d'])
        fit = hybrid_fitness(pars['s'], het, dmi)[0]
        return np.exp(-fit)
    else:
        het = heterosis_effect(t, pars['h'])
        if recessive:
            dmi = dmi_effect_recessive(t, pars['d'])
        elif dominant:
            dmi = dmi_effect_dominant(t, pars['d'])
        else:
            dmi = dmi_effect(t, pars['d'])
        fit = hybrid_fitness(pars['s'], het, dmi)[0]
        return np.exp(fit * (heterogeneous_branching_process_helper(pars, t + 1, n, recessive=recessive,
                                                                    dominant=dominant) - 1))


def adjusted_fixation_probability_heterogeneous_branching_process(pars, t=1, n=800, recessive=False, dominant=False):
    """
    Calculate fixation probability under heterogeneous branching process (Equation S5)
    @param pars: dict, initial heterosis effects ['h'], initial DMI effects ['d'], and selection coefficient ['s']
    @param t: int, current generation >= 1
    @param n: int, final time
    @param recessive: boolean, if recessive-dominance epistasis model should be assumed for DMI effects
    @param dominant: boolean, if dominance-dominance epistasis model should be assumed for DMI effects
    @return: float, fixation probability
    """
    extinction_probability = heterogeneous_branching_process_helper(pars, t, n, recessive, dominant)
    return 1 - extinction_probability


def poisson_offspring(mean):
    """
    Poisson distribution
    @param mean: float, lambda (mean of Poisson distribution)
    @return: Poisson object
    """
    poisson_dist = poisson(mean)
    return poisson_dist
