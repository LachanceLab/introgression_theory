#!/usr/bin/python
import numpy as np
from fitness import heterosis_effect, dmi_effect, dmi_effect_recessive, dmi_effect_dominant


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


def adjusted_probability_of_fixation(N, s, p, alpha):
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
