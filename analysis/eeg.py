import numpy as np
import matplotlib.pyplot as plt

def epochs(data, s_freq, tmin, tmax, onsets, order, desc):

    erp_seconds = (abs(tmin) + tmax)
    erp_ms = int(erp_seconds * 1000)
    
    print("\n\n Epochs : ")
    print("Epoch Len (s) : ", erp_seconds)
    print("Epoch Len (ms) : ", erp_ms)
    
    for idx in range(len(onsets)):
        
        print(f"\nEvent {idx+1}/{len(onsets)} : {order[idx]}-{desc[order[idx]]}")
        onset_seconds = onsets[idx] # Seconds
        print("Epoch Onset (s) : ", onset_seconds)
        print("Epoch Start (s) : ", onset_seconds+tmin)
        print("Epoch End (s) : ", onset_seconds+tmax)
        
        onset = int(onset_seconds * s_freq)
        start = int((onset_seconds + tmin) * s_freq)
        end = int((onset_seconds + tmax) * s_freq)
        epoch_len = end - start

        print("Epoch Len : ", epoch_len)
        print("Epoch Len (s): ", (epoch_len / s_freq))
        print("Epoch Len (ms): ", (epoch_len / s_freq) * 1000)

        epochs = data[:, start:end]
        
        #Baseline correct the epochs
        #baseline = epochs[:, start:onset]
        #baseline_mean = np.mean(baseline, axis=1, keepdims=True)
        #baseline_corrected = epochs - baseline_mean
        
        erp_avg = np.mean(epochs, axis=0)
        
        # Plot ERPs for a single channel (e.g., channel 1)
        time = np.linspace(tmin*1000, tmax*1000, epoch_len)  # Time in ms
        
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(f"Event {idx+1}/{len(onsets)} : {order[idx]}-{desc[order[idx]]}\n" +
                     f"Onset (s) : {onset_seconds}, tmin:{tmin}, tmax:{tmax}, Epoch Len (s):  {(epoch_len / s_freq)}")
        
        for ch in epochs:
            axs[0].plot(time, ch)
        axs[0].set_title("32 Channels Epoch")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Amplitude (µV)")
        
        axs[1].plot(time, erp_avg)
        axs[1].set_title("Averaged Epoch")
        axs[1].set_xlabel("Time (ms)")
        axs[1].set_ylabel("Amplitude (µV)")

        plt.legend()
        plt.show()
        
        pass

    pass

    
    
def event_related_potentials(data, s_freq, tmin, tmax, onsets, order, desc):
    erp_seconds = (abs(tmin) + tmax)
    erp_ms = int(erp_seconds * 1000)
    
    print("\n\n Epochs : ")
    print("Epoch Len (s) : ", erp_seconds)
    print("Epoch Len (ms) : ", erp_ms)
    
    for idx in range(len(onsets)):
        
        print(f"\nEvent {idx+1}/{len(onsets)} : {order[idx]}-{desc[order[idx]]}")
        onset_seconds = onsets[idx] # Seconds
        print("Feature Onset (s) : ", onset_seconds)
        print("Epoch Start (s) : ", onset_seconds+tmin)
        print("Epoch End (s) : ", onset_seconds+tmax)
        
        onset = int(onset_seconds * s_freq)
        start = int((onset_seconds + tmin) * s_freq)
        end = int((onset_seconds + tmax) * s_freq)
        epoch_len = end - start

        print("Epoch Len : ", epoch_len)
        print("Epoch Len (s): ", (epoch_len / s_freq))
        print("Epoch Len (ms): ", (epoch_len / s_freq) * 1000)

        epochs = data[:, start:end]
        
        #Baseline correct the epochs
        #baseline = epochs[:, start:onset]
        #baseline_mean = np.mean(baseline, axis=1, keepdims=True)
        #baseline_corrected = epochs - baseline_mean
        
        erp_avg = np.mean(epochs, axis=0)
        
        # Plot ERPs for a single channel (e.g., channel 1)
        time = np.linspace(tmin*1000, tmax*1000, epoch_len)  # Time in ms
        
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(f"Event {idx+1}/{len(onsets)} : {order[idx]}-{desc[order[idx]]}\n" +
                     f"Onset (s) : {onset_seconds}, tmin:{tmin}, tmax:{tmax}, Epoch Len (s):  {(epoch_len / s_freq)}")
        
        for ch in epochs:
            axs[0].plot(time, ch)
        axs[0].set_title("32 Channels ERP")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Amplitude (µV)")
        
        axs[1].plot(time, erp_avg)
        axs[1].set_title("Averaged ERP")
        axs[1].set_xlabel("Time (ms)")
        axs[1].set_ylabel("Amplitude (µV)")
        
        
        axs[0].axvline(50, linestyle="--", color="r", label="P50")
        axs[0].axvline(100, linestyle="--", color="g", label="N100")
        axs[0].axvline(200, linestyle="--", color="orange", label="P200")
        axs[0].axvline(300, linestyle="--", color="purple", label="P300")
        
        axs[1].axvline(50, linestyle="--", color="r", label="P50")
        axs[1].axvline(100, linestyle="--", color="g", label="N100")
        axs[1].axvline(200, linestyle="--", color="orange", label="P200")
        axs[1].axvline(300, linestyle="--", color="purple", label="P300")

        plt.legend()
        plt.show()
        
        pass

    pass