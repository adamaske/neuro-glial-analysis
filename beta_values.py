import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
duration = 15  # seconds
sampling_rate = 5  # Hz
num_samples = int(duration * sampling_rate)

# Time vector
time = np.linspace(0, duration, num_samples)

# Gaussian distribution parameters
mean = duration / 2  # Center the Gaussian at the middle of the time series
std_dev = duration / 6  # Adjust the standard deviation for desired width

# Gaussian distribution time series
gaussian_distribution_time_series = norm.pdf(time, mean, std_dev)

# Boxcar function
boxcar_function = np.zeros(num_samples)
start_index = int(5 * sampling_rate)
end_index = int(10 * sampling_rate)
boxcar_function[start_index:end_index] = 1

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, gaussian_distribution_time_series)
plt.title("Gaussian Distribution Time Series (5 Hz)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, boxcar_function)
plt.title("Boxcar Function (5-10 seconds)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()

# You can also print the arrays if needed:
# print("Gaussian Distribution Time Series:", gaussian_distribution_time_series)
# print("Boxcar Function:", boxcar_function)

# Y = XB + e
# B = X^TX)^-1XTY
X = boxcar_function
B = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), gaussian_distribution_time_series)

print(B)
