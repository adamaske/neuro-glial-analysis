from eeg import EEG
from fnirs import fNIRS

class EEG_fNIRS:
    def __init__(self, eeg:EEG, fnirs:fNIRS):
        
        self._eeg = eeg
        self._fnirs = fnirs
        
