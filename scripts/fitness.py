#!/usr/bin/python
import numpy as np


def hybrid_fitness(s, h, dmi):
    """
    Compute hybrid fitness
    :param s: float, selection coefficient of tracked allele
    :param h: float, heterosis strength
    :param dmi: float, strength of DMI effects
    :return: float, float, heterozygous hybrid fitness & homozygous hybrid fitness
    """
    w_ab = (1 + s) * (1 + (h + dmi))
    w_bb = (1 + 2 * s) * (1 + (h + dmi))
    if isinstance(w_ab, int):
        if w_ab < 0.0:
            w_ab = 0.0

        if w_bb < 0.0:
            w_bb = 0.0
    else:
        w_ab = np.where(w_ab < 0, 0.0, w_ab)
        w_bb = np.where(w_bb < 0, 0.0, w_bb)

    return w_ab, w_bb


def dmi_effect(t, d):
    """
    :param t: int, np.array time in generations
    :param d: float, DMI effect d <= 0
    :return: int, np.array DMI effect size over time
    """
    assert d <= 0.0, "d={:3f}, DMI effect should be <= 0".format(d)
    if isinstance(t, int):
        if t == 0:
            return 0.0
        else:
            return d * (2 ** -(t - 1) + (1 - 2 ** -(t - 1)) * 2 ** -(t - 1))
    # array
    else:
        decay = np.array([(2 ** -(g - 1) + (1 - 2 ** -(g - 1)) * 2 ** -(g - 1))
                          if g >= 1 else 0.0 for g in range(t.max() + 1)])
        return d * decay


def dmi_effect_dominant(t, d):
    """
    :param t: int, np.array time in generations
    :param d: float, DMI effect d <= 0
    :return: int, np.array DMI effect size over time
    """
    assert d <= 0.0, "d={:3f}, DMI effect should be <= 0".format(d)
    if isinstance(t, int):
        if t == 0:
            return 0.0
        else:
            return d * 2 ** -(t - 1)
    # array
    else:
        decay = np.array([2 ** -(g - 1) if g >= 1 else 0.0 for g in range(t.max() + 1)])
        return d * decay


def dmi_effect_recessive(t, d):
    """
    :param t: int, np.array time in generations
    :param d: float, DMI effect d <= 0
    :return: int, np.array DMI effect size over time
    """
    assert d <= 0.0, "d={:3f}, DMI effect should be <= 0".format(d)
    if isinstance(t, int):
        if t <= 1:
            return 0.0
        else:
            try:
                return d * (2 ** (3 - 2 * t) * (2 ** t - 2))
            except OverflowError:
                return 0.0
    # array
    else:
        decay = np.array([2 ** (3 - 2 * g) * (2 ** g - 2)
                          if g > 1 else 0.0 for g in range(t.max() + 1)])
        return d * decay


def heterosis_effect(t, h):
    """
    :param t: int, np.array time in generations
    :param h: float, heterozygous effect h >= 0
    :return: int, np.array heterozygous effect size over time
    """
    if isinstance(t, int):
        if t == 0:
            return 0
        else:
            return h * (2 ** -(t - 1))
    # array
    else:
        decay = np.array([2 ** -(g - 1) if g > 0 else 0.0 for g in range(t.max() + 1)])
        return h * decay
