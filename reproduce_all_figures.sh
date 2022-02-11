#!/bin/bash
# activate conda env
#conda activate introgression
default_threads=16
threads=${1:-$default_threads}

# Figure 2
python scripts/regimes.py

# Figure 3 (and S2)
python scripts/assess_adjusted_fixation_probability.py -t ${threads} --neutral

# Figure 4
python scripts/assess_impact_on_fixation_probability.py -t ${threads}

# Figure 5
python scripts/analyze_sojourn_times.py

# Figure 6 (and S3 & S4)
python scripts/distribution_of_fitness_effects.py -t ${threads} -b -j

# Figure S1
python scripts/site_frequency_spectrum.py

# Figure S7
python scripts/regimes.py -r

# Figure S8
python scripts/regimes.py -d

# Figures 9
python scripts/assess_adjusted_fixation_probability.py -r -t ${threads}

# Figures S10
python scripts/assess_adjusted_fixation_probability.py --dominant -t ${threads}

# Figure S11
python scripts/assess_impact_on_fixation_probability.py -r -t ${threads}

# Figure S12
python scripts/assess_impact_on_fixation_probability.py -d -t ${threads}

# Figure S13
python scripts/distribution_of_fitness_effects.py -r -t ${threads}

# Figure S14
python scripts/distibution_of_fitness_effects.py -d -t ${threads}

# Figure S15
python scripts/analyze_sojourn_times.py -r

# Figure S16
python scripts/analyze_sojourn_times.py -d