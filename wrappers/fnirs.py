
# Wrappper for mne.io.snirf file
fnirs_types = {0:"WL", 1:"OD", 2:"CC"}

class fNIRS:
    def __init__(self):

        self.lowpass = 0.08
        self.highpass = 0.01
        self.filter_order = 10

        self.snirf = 0
        self.type = "WL"
        pass
    
    def read(self, filepath:str):
        
        self.filepath = filepath
        self.snirf = 0

        return self
    
    def write(self, filepath:str):

        pass


    