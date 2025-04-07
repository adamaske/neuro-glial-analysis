import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def calculate_instantaneous_phase(time_series):
    """Calculates the instantaneous phase clipped to [0, 2*pi]."""
    analytic_signal = hilbert(time_series)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    return np.mod(instantaneous_phase, 2 * np.pi)

# Parameters
sampling_rate = 1000  # Hz
duration = 2  # seconds
time = np.arange(0, duration, 1 / sampling_rate)

# Generate two signals with slightly different frequencies
frequency1 = 5  # Hz
frequency2 = 5  # Hz
signal1 = np.sin(2 * np.pi * frequency1 * time)
signal2 = np.sin(2 * np.pi * frequency2 * time)

# Calculate instantaneous phases
phase1 = calculate_instantaneous_phase(signal1)
phase2 = calculate_instantaneous_phase(signal2)

window_sizee = 0.2 # seconds
window_step = 0.2

phase_differences = []
for i, t in enumerate(time):

    theta_a = phase1[i]
    theta_b = phase2[i]

    theta = theta_a - theta_b

    
    phase_differences.append(np.exp(1j * theta))
    

m = np.abs(np.mean(phase_differences))
print("|m| : ", m)

exit()

# Plot the signals
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, signal1, label=f'{frequency1} Hz', color="red")
plt.plot(time, signal2, label=f'{frequency2} Hz', color="green")
plt.title('Two Signals')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Plot the instantaneous phases
plt.subplot(2, 1, 2)
plt.plot(time, phase1, label=f'{frequency1} Hz Phase', color="red")
plt.plot(time, phase2, label=f'{frequency2} Hz Phase', color="green")
plt.title('Instantaneous Phases (clipped)')
plt.xlabel('Time (s)')
plt.ylabel('Phase (radians)')
plt.legend()

plt.tight_layout()
plt.show()
exit()
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def instantaneous_phase_frequency_plot(time_series_A, sampling_rate, title="Instantaneous Phase, Frequency, and Signal"):
    """
    Calculates and plots the instantaneous phase and frequency of a time series,
    along with a phase wrapping degree plot and the original signal.

    Args:
        time_series_A: NumPy array representing the time series.
        sampling_rate: Sampling rate of the time series (Hz).
        title: Title of the plot.
    """

    analytic_signal = hilbert(time_series_A)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_phase = np.mod(instantaneous_phase, 2 * np.pi)
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * sampling_rate

    time = np.arange(len(time_series_A)) / sampling_rate

    plt.figure(figsize=(12, 10)) #increased figure size

    # Plot Original Signal
    plt.subplot(3, 1, 1)
    plt.plot(time, time_series_A)
    plt.title(f"{title} - Original Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot Instantaneous Phase
    plt.subplot(3, 1, 2)
    plt.plot(time, instantaneous_phase)
    plt.title(f"{title} - Instantaneous Phase")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")

    # Plot Instantaneous Frequency
    plt.subplot(3, 1, 3)
    plt.plot(time[1:], instantaneous_frequency)
    plt.title(f"{title} - Instantaneous Frequency")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.show()

# Example Usage:
# Generate a sample time series
sampling_rate = 5  # Hz
duration = 20
time = np.arange(0, duration, 1/sampling_rate)
frequency = 0.12 # Hz
time_series_A = np.sin(2 * np.pi * frequency * time) + np.random.randn(len(time)) * 0.1

# Plot instantaneous phase and frequency
instantaneous_phase_frequency_plot(time_series_A, sampling_rate)