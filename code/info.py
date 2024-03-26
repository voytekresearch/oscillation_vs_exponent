"""
Dataset details
"""

PATIENTS = ['pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16',
            'pat17','pat19','pat20','pat21','pat22']
FS = 512 # iEEG sampling frequency
TMIN = -1.5 # epoch start time
MATERIALS = ['words', 'faces'] # experimental blocks (stimulus type) 
MEMORY = ['hit', 'miss'] # behavior (memory performance)
FELLNER_BANDS = {'theta' : [2, 5], 'alpha' : [8, 20], 'gamma' : [50, 90]} # oscillation bands used in Fellner et al. 2016
