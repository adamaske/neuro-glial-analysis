import h5py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def bandpass_filter(data, sfreq, low_freq=1, high_freq=100, order=4):
    """Apply a bandpass filter to EEG data."""
    nyquist = 0.5 * sfreq
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def notch_filter(data, sfreq, freqs=[50, 60]):
    """Apply notch filters at specified frequencies."""
    for freq in freqs:
        b, a = iirnotch(freq, 30, sfreq)
        data = filtfilt(b, a, data, axis=-1)
    return data

def process_eeg_hdf5(file_path, output_path):
    """Load EEG data from HDF5, filter it, and save the result."""
    
    with h5py.File(file_path, 'r') as f:
        sfreq = f['sfreq'][()]  # Sampling frequency
        eeg_data = f['eeg_data'][()]  # EEG data (channels x time)

    # Apply bandpass and notch filters
    filtered_data = bandpass_filter(eeg_data, sfreq)
    filtered_data = notch_filter(filtered_data, sfreq)

    # Save filtered EEG data
    with h5py.File(output_path, 'w') as f_out:
        f_out.create_dataset('eeg_data', data=filtered_data)
        f_out.create_dataset('sfreq', data=sfreq)  # Save sampling frequency as well

    print(f"Filtered EEG data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    input_file = "raw_eeg.hdf5"
    output_file = "filtered_eeg.hdf5"
    process_eeg_hdf5(input_file, output_file)
