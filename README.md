# Oscillations and aperiodic activity: Evidence for dynamic changes in both during memory encoding
This repository provides code for analyzing stimulus-evoked dynamics of oscillatory and aperiodic electrophysiological activity in the dataset detailed below.

## Dataset
The data analyzed are openly available on the Open Science Framework repository (https://osf.io/3csku/).

This dataset is associated with the following publication:
Fellner M-C, Gollwitzer S, Rampp S, Kreiselmeyr G, Bush D, Diehl B, et al. Spectral fingerprints or spectral tilt? Evidence for distinct oscillatory signatures of memory formation. Knight RT, editor. PLOS Biol. 2019;17: e3000403. doi:10.1371/journal.pbio.3000403

## Requirements
numpy  
scipy  
pandas  
matplotlib  
seaborn  
statsmodels  
pingouin  
scikit-learn  
mne  
joblib  
pymatreader  
pyvista  
nilearn  
neurodsp  
specparam  

## Pipeline
To reproduce the figures associated with the manuascript, navigate to the base directory of the repository and execute ```make all```. The scripts can also be executed sequentially in the order listed in the ```Makefile```.