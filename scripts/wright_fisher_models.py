#!/usr/bin/env python
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


def wright_fisher_poisson_decay_hfe(p0, pars, max_generations=np.inf, genome_size=3200000000):
    """
    Wright Fisher model using fitness function accounting for hybrid effects
    :param p0: float, initials frequency of allele
    :param pars: dict, must hold parameters h, d, s, and N
    :param max_generations: int, maximal number of generations, default=infinity
    :param genome_size: int, genome size used to determine number of trials
    :return: list allele frequency over time
    """
    # initialize
    p = p0
    t = 1
    # keep track of genotypes
    genotypes = np.zeros((2, pars["N"]))
    genotypes[0, 0] = 1
    # keep track of fitness
    fitness = np.ones(pars['N'])
    w_ab = (1 + pars['s']) * (1 + pars['h'] + pars['d'])
    fitness[0] = w_ab
    # keep track of amount of donor DNA in an individual
    donor_dna = np.zeros(pars['N'])
    donor_dna[0] = 1
    frequency = [0]
    while p != 0.0 and p != 1.0 and t < max_generations:
        # pick parents
        parent_A = np.random.choice(np.arange(0, genotypes.shape[1]), size=genotypes.shape[1],
                                    p=fitness / fitness.sum())
        parent_B = np.random.choice(np.arange(0, genotypes.shape[1]), size=genotypes.shape[1],
                                    p=fitness / fitness.sum())

        # avoid selfing
        while np.any(parent_A == parent_B):
            parent_B = np.where(parent_A == parent_B, np.random.choice(np.arange(0, genotypes.shape[1]),
                                                                       size=1, p=fitness / fitness.sum()),
                                parent_B)
        # pick allele
        allele_A = np.random.randint(0, 2, size=genotypes.shape[1])
        allele_B = np.random.randint(0, 2, size=genotypes.shape[1])

        # genotype of genotypes at marker locus
        new_genotypes = np.stack([genotypes[allele_A, parent_A], genotypes[allele_B, parent_B]])
        # compute genotypes next generation
        het_individuals = np.where(new_genotypes.sum(axis=0) == 1)[0]
        hom_individuals = np.where(new_genotypes.sum(axis=0) == 2)[0]

        # compute fraction of donor dna
        # one recombination event every 100Mb --> 32 events in human genome
        # chance of including certain segment is 0.5
        # divide by 32 to get fraction of donor segments
        trials = int(genome_size / 100000000)
        fraction_hybrid_A = donor_dna[parent_A] * (np.random.binomial(trials, p=0.5, size=genotypes.shape[1]) / trials)
        fraction_hybrid_B = donor_dna[parent_B] * (np.random.binomial(trials, p=0.5, size=genotypes.shape[1]) / trials)

        # assume donor dna is not inherited without the marker allele. This is valid for large populations
        # as we deal with in the main text. However, it is unrealistic in small populations,
        # and my inflate the effect of HFEs. To guarantee comparability of the results,
        # I need to synthetically implement this assumption.
        fraction_hybrid_A = np.where(genotypes[allele_A, parent_A] == 1, fraction_hybrid_A, 0)
        fraction_hybrid_B = np.where(genotypes[allele_B, parent_B] == 1, fraction_hybrid_B, 0)
        donor_dna = fraction_hybrid_A + fraction_hybrid_B
        donor_dna[np.where(new_genotypes.sum(axis=0) == 0)[0]] = 0
        donor_dna = np.where(donor_dna > 1, 1, donor_dna)

        # base fitness is one,
        fitness = np.ones_like(fitness)
        # heterozygote fitness
        if het_individuals.shape[0] > 0:
            het = pars['h'] * donor_dna[het_individuals]
            dmi = pars['d'] * (donor_dna[het_individuals] + donor_dna[het_individuals] * (1 - donor_dna[het_individuals]))
            w_ab = (1 + pars['s']) * (1 + het + dmi)
            fitness[het_individuals] = w_ab

        # homozygote fitness
        if hom_individuals.shape[0] > 0:
            het = pars['h'] * donor_dna[hom_individuals]
            dmi = pars['d'] * (donor_dna[hom_individuals] + donor_dna[hom_individuals] * (1 - donor_dna[hom_individuals]))
            w_bb = (1 + 2 * pars['s']) * (1 + het + dmi)
            fitness[hom_individuals] = w_bb

        # can't be less than 0
        fitness = np.where(fitness < 0, 0, fitness)
        # update frequency
        genotypes = new_genotypes
        p = genotypes.sum() / np.prod(genotypes.shape)
        frequency.append(p)
        t += 1

    return frequency
