"""
Dataset details
"""

PATIENTS = ['pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16',
            'pat17','pat19','pat20','pat21','pat22']
FS = 512 # iEEG sampling frequency
TMIN = -1.5 # epoch start time
MATERIALS = ['words', 'faces'] # experimental blocks (stimulus type) 
MEMORY = ['hit', 'miss'] # behavior (memory performance)