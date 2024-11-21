import matplotlib.pyplot as plt


from mne.io.snirf._snirf import RawSNIRF
def visualize_snirf(snirf):

    pass

def plot_raw_channels(snirf:RawSNIRF, channels=None):
    """
    Display fNIRS channels
    Args:
        snirf (RawSNIRF) : snirf object.
        channels (int or list) : Channel index or list of indices
    """

    print(snirf.info)

    plt.show()

    return