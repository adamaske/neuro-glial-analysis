##
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import correlate, coherence, hilbert
import seaborn as sns
def pearson_correlation(channel_data):

    r_matrix = np.zeros((channel_data.shape[0], channel_data.shape[0]))
    p_matrix = np.zeros((channel_data.shape[0], channel_data.shape[0]))
    for i, ch1 in enumerate(channel_data):
        for j, ch2 in enumerate(channel_data):
            r, p = pearsonr(ch1, ch2)
            r_matrix[i, j] = r
            p_matrix[i, j] = p
    return r_matrix, p_matrix

def cross_correlation(channel_data):
    
    r_matrix = np.zeros((channel_data.shape[0], channel_data.shape[0]))
    for i, ch1 in enumerate(channel_data):
        for j, ch2 in enumerate(channel_data):
            
            Cxy = correlate(ch1, ch2, mode='full')
            Cxy_normalized = Cxy / (np.sqrt(np.sum(ch1**2)) * np.sqrt(np.sum(ch2**2)))
            r_matrix[i, j] = np.max(np.abs(Cxy_normalized))
    return r_matrix

def coherence_correlation(channel_data, sampling_rate, segment_length, f_low, f_high):

    r_matrix = np.zeros((channel_data.shape[0], channel_data.shape[0]))
    for i, ch1 in enumerate(channel_data):
        for j, ch2 in enumerate(channel_data):
            f, Cxy = coherence(ch1, ch2, fs=sampling_rate, nperseg=segment_length)
            band_indices = np.where((f >= f_low) & (f <= f_high))[0]
            r_matrix[i, j] = np.mean(Cxy[band_indices]) # TODO : How to handle this?

    return r_matrix



def calculate_instantaneous_phase(time_series):
    """Calculates the instantaneous phase clipped to [0, 2*pi]."""
    analytic_signal = hilbert(time_series)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    return np.mod(instantaneous_phase, 2 * np.pi), analytic_signal

def phase_clustering(channel_data):

    m_matrix = np.zeros((channel_data.shape[0], channel_data.shape[0]))
    for i, ch1 in enumerate(channel_data):
        for j, ch2 in enumerate(channel_data):

            phase_ch1, analytical_ch1 = calculate_instantaneous_phase(ch1) # theta_x
            phase_ch2, analytical_ch2 = calculate_instantaneous_phase(ch2) # theta_y

            #phase difference
            phase_diff = phase_ch1 - phase_ch2

            avg_phase_diff = np.mean(np.exp(1j * phase_diff))
            phase_diff_m = np.abs(avg_phase_diff)
            m_matrix[i, j] = phase_diff_m

            if True:
                continue
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
            # Plot Ch1 Phase on Unit Circle
            #-> Avg vector in a different color

            fig.suptitle(f"Channel {i} & {j}")
            plt.subplot(1, 3, 1)
            ch1_circle = plt.Circle((0, 0), 1, color='black', fill=False)
            ax1.add_artist(ch1_circle)
            for k, phase in enumerate(phase_ch1):
                # Vectors
                x1, y1 = np.cos(phase_ch1[k]), np.sin(phase_ch1[k])
                ax1.arrow(0, 0, x1, y1, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
             
            avg_ch1 = np.mean(np.exp(1j * phase_ch1))
            avg_ch1_m = np.abs(avg_ch1)
            ax1.arrow(0, 0, np.real(avg_ch1), np.imag(avg_ch1), head_width=0.1, head_length=0.1, fc='green', ec='green', label=f'm : {avg_ch1_m:.2f}')

            
            # Plot Ch2 Phase on Unit Circle
            #-> Avg vector in a different color
            ch2_circle = plt.Circle((0, 0), 1, color='black', fill=False)
            ax2.add_artist(ch2_circle)
            for k, phase in enumerate(phase_ch2):
                # Vectors
                x1, y1 = np.cos(phase_ch2[k]), np.sin(phase_ch2[k])
                ax2.arrow(0, 0, x1, y1, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            
            avg_ch2 = np.mean(np.exp(1j * phase_ch2))
            avg_ch2_m = np.abs(avg_ch2)
            ax2.arrow(0, 0, np.real(avg_ch2), np.imag(avg_ch2), head_width=0.1, head_length=0.1, fc='green', ec='green', label=f'm : {avg_ch2_m:.2f}')

            # Plot The Phase Difference on Unit Circle
            #-> Avg in a different color
            diff_circle = plt.Circle((0, 0), 1, color='black', fill=False)
            ax3.add_artist(diff_circle)

            for k, phase in enumerate(phase_diff):
                # Vectors
                x1, y1 = np.cos(phase_diff[k]), np.sin(phase_diff[k])
                ax3.arrow(0, 0, x1, y1, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            
            ax3.arrow(0, 0, np.real(avg_phase_diff), np.imag(avg_phase_diff), head_width=0.1, head_length=0.1, fc='green', ec='green', label=f'm : {phase_diff_m:.2f}')

            # Set ylim and xlim
            for ax in [ax1, ax2, ax3]:
                ax.set_ylim(-1.2, 1.2)
                ax.set_xlim(-1.2, 1.2)
                ax.set_aspect('equal', adjustable='box') # make the circle look circular
                ax.legend()
            ax1.set_title("Channel 1")
            ax2.set_title("Channel 2")
            ax3.set_title("Phase Difference")
            plt.show()
    return m_matrix

def phase_lag_index(channel_data):

    pli_matrix = np.zeros((channel_data.shape[0], channel_data.shape[0]))
    for i, ch1 in enumerate(channel_data):
        for j, ch2 in enumerate(channel_data):

            phase_ch1, analytical_ch1 = calculate_instantaneous_phase(ch1) # theta_x
            phase_ch2, analytical_ch2 = calculate_instantaneous_phase(ch2) # theta_y

            phase_diff = phase_ch1 - phase_ch2

            pli = np.abs(np.mean(np.sign(np.imag(np.exp(1j * phase_diff)))))
            pli_matrix[i, j] = pli

    return pli_matrix

def weighted_pli(channel_data):

    wpli_matrix = np.zeros((channel_data.shape[0], channel_data.shape[0]))
    for i, ch1 in enumerate(channel_data):
        for j, ch2 in enumerate(channel_data):

            phase_ch1, analytical_ch1 = calculate_instantaneous_phase(ch1) # theta_x
            phase_ch2, analytical_ch2 = calculate_instantaneous_phase(ch2) # theta_y

            phase_diff = phase_ch1 - phase_ch2

            imag_phase_diff = np.imag(np.exp(1j * phase_diff))
            numerator = np.abs(np.mean(imag_phase_diff))
            denominator = np.mean(np.abs(imag_phase_diff))

            if denominator != 0:
                wpli = numerator / denominator
            else:
                wpli = 0 # Avoid division by zero

            wpli_matrix[i, j] = wpli

    return wpli_matrix

def plot_r_matrix(r_matrix, channel_names, title):

    sns.heatmap(r_matrix, 
                annot=False, 
                cmap='RdBu_r', 
                vmin=-1, 
                vmax=1, 
                xticklabels=channel_names, 
                yticklabels=channel_names)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
  

def composite_correlation(channel_data, channel_names, sampling_rate, segment_length:float|None, f_low, f_high):
    """
    Calculates and visualizes various correlation measures and their composite.

    Args:
        channel_data (np.ndarray): 2D array where rows are channels and columns are time points.
        channel_names (list of str): List of channel names, one name per row of channel_data.
        sampling_rate (float): Sampling rate of the channels in Hz.
        segment_length (float or None): Segment length for coherence calculation in seconds.
                                       If None, the entire signal is used.
        f_low (float): Lower frequency bound for coherence calculation in Hz.
        f_high (float): Upper frequency bound for coherence calculation in Hz.

    Returns:
        np.ndarray: The mean of all correlation matrices (composite correlation).
    """
    r, p = pearson_correlation(channel_data)
    r_cross = cross_correlation(channel_data)
    r_coh = coherence_correlation(channel_data, sampling_rate, segment_length, f_low, f_high)
    r_pc = phase_clustering(channel_data)
    r_pli = phase_lag_index(channel_data)
    r_wpli = weighted_pli(channel_data)
    r_mean =  np.mean([r, r_cross, r_coh, r_pc, r_pli, r_wpli], axis=0)

    plt.figure(figsize=(14, 9))
    plt.subplot(2, 3, 1)
    plot_r_matrix(r, channel_names, "Pearson Correlation")
    plt.subplot(2, 3, 2)
    plot_r_matrix(r_cross, channel_names, "Cross-Correlation")
    plt.subplot(2, 3, 3)
    plot_r_matrix(r_coh, channel_names, "Coherence")
    plt.subplot(2, 3, 4)
    plot_r_matrix(r_pc, channel_names, "Phase Clustering")
    plt.subplot(2, 3, 5)
    plot_r_matrix(r_pli, channel_names, "Phase Lag Index")
    plt.subplot(2, 3, 6)
    plot_r_matrix(r_wpli, channel_names, "Weighted PLI")
    plt.figure(figsize=(6, 5))
    plot_r_matrix(r_mean, channel_names, "Composite Correlation")

    return r_mean

# Function to generate a channel with a given fundamental frequency and noise
def generate_channel(freq, sampling_rate, duration, noise_level=1.0):
    time = np.arange(0, duration, 1/sampling_rate)
    signal = np.sin(2 * np.pi * freq * time)
    noise = noise_level * np.random.randn(len(time))
    return signal + noise

if False:
    # Parameters
    num_channels = 5
    sampling_rate = 100  # Samples per second
    duration = 1.0  # Seconds
    fundamental_frequencies = np.array([15.0, 15.25, 15.5, 15.75, 16.0])  # Different fundamental frequencies for each channel
    noise_level = 0.7  # Adjust noise level to control the randomness
    
    # Generate the channels
    channels = np.zeros((num_channels, int(sampling_rate * duration)))
    for i in range(num_channels):
        channels[i, :] = generate_channel(fundamental_frequencies[i], sampling_rate, duration, noise_level)
    
    ch_names = ["ch1", "ch2", "ch3", "ch4", "ch5"]
    
    r, p = pearson_correlation(channels)
    r_cross = cross_correlation(channels)
    r_coh = coherence_correlation(channels, sampling_rate, 16, 10, 20)
    r_pc = phase_clustering(channels)
    r_pli = phase_lag_index(channels)
    r_wpli = weighted_pli(channels)
    r_mean =  np.mean([r, r_cross, r_coh, r_pc, r_pli, r_wpli], axis=0)
    
    plt.figure(figsize=(14, 9))
    #plt.title(f"Functional Connectivity : [{fundamental_frequencies}] Hz")
    plt.subplot(2, 3, 1)
    plot_r_matrix(r, ch_names, "Pearson Correlation")
    plt.subplot(2, 3, 2)
    plot_r_matrix(r_cross, ch_names, "Cross-Correlation")
    plt.subplot(2, 3, 3)
    plot_r_matrix(r_coh, ch_names, "Coherence")
    plt.subplot(2, 3, 4)
    plot_r_matrix(r_pc, ch_names, "Phase Clustering")
    plt.subplot(2, 3, 5)
    plot_r_matrix(r_pli, ch_names, "Phase Lag Index")
    plt.subplot(2, 3, 6)
    plot_r_matrix(r_wpli, ch_names, "Weighted PLI")
    plt.figure(figsize=(6, 5))
    plot_r_matrix(r_mean, ch_names, "Composite Correlation")
    
    plt.figure(figsize=(10, 8))  # Adjust figure size as needed
    
    time = np.arange(0, duration, 1 / sampling_rate)
    
    for i, channel in enumerate(channels):
        plt.subplot(num_channels, 1, i + 1)  # Create subplots
        plt.plot(time, channel)
        plt.title(f"{ch_names[i]} @ {fundamental_frequencies[i]:.2f} Hz")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
    
    plt.tight_layout()  # Adjust layout to prevent overlapping titles
    plt.show()