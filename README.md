# Oscillations and aperiodic activity: Evidence for dynamic changes in both during memory encoding
This repository provides code for analyzing stimulus-evoked dynamics of oscillatory and aperiodic electrophysiological activity in the dataset detailed below.

## Dataset
The data analyzed are openly available on the Open Science Framework repository (https://osf.io/3csku/).

This dataset is associated with the following publication:
Fellner M-C, Gollwitzer S, Rampp S, Kreiselmeyr G, Bush D, Diehl B, et al. Spectral fingerprints or spectral tilt? Evidence for distinct oscillatory signatures of memory formation. Knight RT, editor. PLOS Biol. 2019;17: e3000403. doi:10.1371/journal.pbio.3000403

## Requirements
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)
- [scipy](https://github.com/scipy/scipy)
- [matplotlib](https://github.com/matplotlib/matplotlib)
- [pymatreader](https://pypi.org/project/pymatreader/)
 - [mne](https://github.com/mne-tools/mne-python)
 - [fooof](https://github.com/fooof-tools/fooof)
- [neurodsp](https://github.com/neurodsp-tools/neurodsp)

## Pipeline
To reproduce the analyses, sequencially execute the scripts in ```code```. All figures can then be reproduced in the Jupyter notebook: /notebooks/manuscript_figures.ipynb