from mne.io.snirf._snirf import RawSNIRF
import matplotlib.pyplot as plt
def display_snirf(snirf:RawSNIRF) -> None:
    
    snirf.plot()
    plt.show()
    
    return None