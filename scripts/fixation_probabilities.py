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


def adjusted_fixation_probability_homogeneous_branching_process(s, alpha):
    """
    Compute adjusted fixation probability using probability generating functions under a branching process
    @param s: float, selection coefficient
    @param alpha: float, compounded effect size of HFEs
    @return: float, fixation probability
    """
    xmin = 0
    xmax = 1e6
    dx = 1
    xvals = np.arange(xmin, xmax + dx, dx)
    # get pmf
    mean = 1 + s * alpha
    poisson_dist = poisson_offspring(mean)
    poisson_dist_pmf = poisson_dist.pmf(xvals)
    # generating function
    extinction = lambda u: np.sum(poisson_dist_pmf * u ** xvals) - u
    # solve
    ustar = fsolve(extinction, 0)
    return 1 - min(ustar)


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
        return np.exp(fit * (heterogeneous_branching_process_helper(pars, t + 1, n) - 1))


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


# def p_extinction_bp_recursion_helper(k, pars, max_iter=1000, tol=1e-10):
#     """
#     Recursively compute extinction probability under branching process
#     @param k: np.array, possible number of offsprings
#     @param pars: dict, parameters for initial strength of heterosis (h) and DMI effects (d), and selection coefficient (s)
#     @param max_iter: int, max_number of iterations
#     @param tol: float, tolerance if change in extinction probability is less than threshold the loop is interrupted
#     @return: float, extinction probability
#     """
#     mem = np.zeros(max_iter)
#     t = 1
#     het = heterosis_effect(np.arange(1, max_iter + 1), pars['h'])[1:]
#     dmi = dmi_effect(np.arange(1, max_iter + 1), pars['d'])[1:]
#     # heterozygote fitness
#     fit = 1 + pars['s'] * np.prod(1 + het[:t] + dmi[:t])
#     poisson_dist = poisson_offspring(fit)
#     p_k = poisson_dist.pmf(k)
#     mem[t - 1] = p_k[0]
#     t += 1
#     while t <= max_iter:
#         #         het = heterosis_effect(t, pars['h'])
#         #         dmi = dmi_effect(t, pars['d'])
#         # heterozygote fitness
#         fit = 1 + pars['s'] * np.prod(1 + het[:t] + dmi[:t])
#         poisson_dist = poisson_offspring(fit)
#         p_k = poisson_dist.pmf(k)
#         mem[t - 1] = np.sum(p_k * mem[t - 2] ** k)
#         if np.abs(mem[t - 2] - mem[t - 1]) < tol:
#             break
#         t += 1
#     if t > max_iter:
#         t = max_iter
#     return mem[t - 1]
#
#
# def adjusted_fixation_probability_branching_process_recursion(pars):
#     """
#     Compute fixation probability from branching process recursively
#     @param pars: dict, parameters for initial strength of heterosis (h) and DMI effects (d), and selection coefficient (s)
#     @return: float, fixation probability
#     """
#     xmin = 0
#     xmax = 1e6
#     dx = 1
#     k = np.arange(xmin, xmax + dx, dx)
#     extinction_prob = p_extinction_bp_recursion_helper(k, pars)
#     return 1 - extinction_prob


def poisson_offspring(mean):
    """
    Poisson distribution
    @param mean: float, lambda (mean of Poisson distribution)
    @return: Poisson object
    """
    poisson_dist = poisson(mean)
    return poisson_dist
