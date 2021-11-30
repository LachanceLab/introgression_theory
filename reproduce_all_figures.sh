#!/bin/bash
# activate conda env
#conda activate introgression
default_threads=16
threads=${1:-$default_threads}

# Figure 2
python scripts/regimes.py

# Figure 3
python scripts/assess_impact_on_fixation_probability.py -t ${threads}

# Figure 4 (and S9 & S10)
python scripts/assess_adjusted_fixation_probability.py -t ${threads} -e

# Figure 5
python scripts/analyze_sojourn_times.py

# Figure 6 (and S15 & S16)
python scripts/distribution_of_fitness_effects.py -t ${threads} -b -j

# Figure S3
python scripts/regimes.py -r

# Figure S4
python scripts/regimes.py -d

# Figure S5
python scripts/assess_impact_on_fixation_probability.py -r -t ${threads}

# Figure S6
python scripts/assess_impact_on_fixation_probability.py -d -t ${threads}

# Figures S7 & S11
python scripts/assess_adjusted_fixation_probability.py -r -t ${threads}

# Figures S8 & 12
python scripts/assess_adjusted_fixation_probability.py --dominant -t ${threads}

# Figure S13
python scripts/analyze_sojourn_times.py -r

# Figure S14
python scripts/analyze_sojourn_times.py -d

# Figure S17
python scripts/distribution_of_fitness_effects.py -r -t ${threads}

# Figure S18
python scripts/distribution_of_fitness_effects.py -d -t ${threads}
