import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import stft

def epochs(data, s_freq, tmin, tmax, onsets, order, desc):

    epoch_seconds = (abs(tmin) + tmax)
    epoch_ms = int(epoch_seconds * 1000)
    
    for idx in range(len(onsets)):
        onset_seconds = onsets[idx] # Seconds
        
        onset = int(onset_seconds * s_freq)
        start = int((onset_seconds + tmin) * s_freq)
        end = int((onset_seconds + tmax) * s_freq)
        epoch_len = end - start
        epochs = data[:, start:end]
        
        #baseline = epochs[:, start:onset]
        #baseline_mean = np.mean(baseline, axis=1, keepdims=True)
        #baseline_corrected = epochs - baseline_mean
    
        epoch_avg = np.mean(epochs, axis=0)
        
        
        time = np.linspace(tmin*1000, tmax*1000, epoch_len)
        
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(f"Epoch {idx+1}/{len(onsets)} : {order[idx]}-{desc[order[idx]]}\n" +
                     f"Onset (s) : {onset_seconds}, tmin:{tmin}, tmax:{tmax}, Epoch Len (s):  {(epoch_len / s_freq)}")
        
        for ch in epochs:
            axs[0].plot(time, ch)
        axs[0].set_title("32 Channels")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Amplitude (µV)")
        axs[0].axvline(0, linestyle="--", color="r", label="Onset")
        axs[0].legend()
        
        axs[1].plot(time, epoch_avg)
        axs[1].set_title("Averaged")
        axs[1].set_xlabel("Time (ms)")
        axs[1].set_ylabel("Amplitude (µV)")
        axs[1].axvline(0, linestyle="--", color="r", label="Onset")
        axs[1].legend()
        plt.show()
    
def event_related_potentials(data, s_freq, P, N, onsets, order, desc):
    erp_seconds = 1
    erp_ms = int(erp_seconds * 1000)
    
    for idx in range(len(onsets)):
        
        onset = onsets[idx] # Seconds
        start = int(onset * s_freq)
        end = int((onset + erp_seconds) * s_freq)

        epochs = data[:, start:end]
        epoch_avg = np.mean(epochs, axis=0)
        
        # TODO : Baseline correction -> take mean of signal seconds before onset and subtract from epoch
        
        t = np.linspace(0, 1, end-start)  # Time in ms
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(f"Event {idx+1}/{len(onsets)} : {order[idx]}-{desc[order[idx]]}\n" +
                     f"Onset (s) : {onset} Epoch Len (s):  {((end-start) / s_freq)}")
        
        for ch in epochs:
            axs[0].plot(t, ch)
        axs[0].set_title("32 Channels ERP")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Amplitude (µV)")
        
        axs[1].plot(t, epoch_avg)
        axs[1].set_title("Averaged ERP")
        axs[1].set_xlabel("Time (ms)")
        axs[1].set_ylabel("Amplitude (µV)")
        
        colors_p = ['r', 'g', 'b', 'm', 'c'] 
        colors_n = ['orange', 'purple', 'brown', 'gray', 'pink'] 
        for i, p in enumerate(P):
            color = colors_p[i % len(colors_p)] 
            axs[0].axvline(p / erp_ms, linestyle="--", color=color, label=f"P{p}")
            axs[1].axvline(p / erp_ms, linestyle="--", color=color, label=f"P{p}")

        for i, n in enumerate(N):
            color = colors_n[i % len(colors_n)] 
            axs[0].axvline(n / erp_ms, linestyle="--", color=color, label=f"N{n}")
            axs[1].axvline(n / erp_ms, linestyle="--", color=color, label=f"N{n}")
            
        axs[0].legend()
        axs[1].legend()
        plt.show()
        pass

    pass

def short_time_fourier_transform(time_series, s_freq):
    
    freqs, t, m = stft(time_series, s_freq, window='gaussian', nperseg=256, noverlap=128)
    return 

def multi_channel_stft(data, s_freq):
    
    stft_results = [short_time_fourier_transform(channel, s_freq) for channel in data]

    return stft_results


def continuous_wavelet_transform(time_series, s_freq):
    
    pass
def multi_channel_cwt(data, s_freq):


    pass
    
