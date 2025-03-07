import numpy as np
import matplotlib.pyplot as plt

from hrf import chrf

t = np.linspace(0, 20, 100)
hrf = chrf(t, 6, 16, 1, 1, 1/6)

#Parameters
duration = 15  # seconds
sampling_rate = 5  # Hz
num_samples = int(duration * sampling_rate)
time = np.linspace(0, duration, num_samples)
mean = 0
std_dev = 1
# Gaussian time series
gaussian_time_series = np.exp(-0.5 * ((time - mean) / std_dev)**2)

# Boxcar function (same as before)
boxcar_function = np.zeros(num_samples)
start_index = int(5 * sampling_rate)
end_index = int(10 * sampling_rate)
boxcar_function[start_index:end_index] = 1

# Y = BX + e
# B = (X^T*X)^-1 * X^T * Y
def b_value(y, x):
    
    X_transpose = y.T
    X_transpose_X = np.matmul(X_transpose, x)
    try:
        x_transpose_x_inverse = np.linalg.inv(X_transpose_X)
    except np.linalg.LinAlgError:
        return 0 # 
    X_transpose_Y = np.matmul(X_transpose, y)
    b = np.matmul(x_transpose_x_inverse, X_transpose_Y)
    return b
    
    
# Reshape X to be a column vector (necessary for matrix multiplication)
X = boxcar_function.reshape(-1, 1) # or boxcar_function[:, np.newaxis]

# Reshape Y to be a column vector.
Y = gaussian_time_series.reshape(-1, 1)

# Sliding window parameters
window_size = int(2 * sampling_rate)  # 2 seconds window
step_size = int(1 * sampling_rate)  # 1 second step


# Sliding window parameters
window_size = int(2 * sampling_rate)  # 2 seconds window
step_size = int(1 * sampling_rate)  # 1 second step

# Calculate beta values for each window
beta_values = np.zeros(num_samples)  # Initialize with NaNs
window_times = []

for i in range(0, num_samples - window_size + 1, step_size):
    window_start = i
    window_end = i + window_size

    window_y = gaussian_time_series[window_start:window_end]
    window_x = boxcar_function[window_start:window_end]

    # Reshape for linear regression
    window_x = window_x.reshape(-1, 1)
    window_y = window_y.reshape(-1, 1)

    def beta_value(x, y):
        # Calculate beta
        x_transpose = x.T
        x_transpose_x = np.dot(x_transpose, x)
        try:
            x_transpose_x_inverse = np.linalg.inv(x_transpose_x)
        except np.linalg.LinAlgError:
            return 0  # Skip this window if matrix is singular

        x_transpose_y = np.dot(x_transpose, y)
        return np.dot(x_transpose_x_inverse, x_transpose_y)[0, 0] #extract scalar beta
    
    beta = beta_value(window_x, window_y)
    center_index = window_start + window_size // 2
    beta_values[center_index] = beta #add beta value to the correct index

# Calculate B (beta)
X_transpose = X.T
X_transpose_X = np.matmul(X_transpose, X)
X_transpose_X_inverse = np.linalg.inv(X_transpose_X)
X_transpose_Y = np.matmul(X_transpose, Y)
B = np.matmul(X_transpose_X_inverse, X_transpose_Y)

# Print the beta value
print("Beta (B):", B)

# Optional: Predict Y using the calculated beta
Y_predicted = np.matmul(X, B)

# Perform convolution
convolution_result = np.convolve(gaussian_time_series, boxcar_function, mode='same')



# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(4, 1, 1)
plt.plot(time, gaussian_time_series)
plt.title("Gaussian Time Series (5 Hz)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(time, boxcar_function)
plt.title("Boxcar Function (5-10 seconds)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 2, 1)
plt.plot(time, beta_values, label="Sliding Window Beta Values")
plt.title("Sliding Window Beta Values")
plt.xlabel("Time (seconds)")
plt.ylabel("Beta Value")
plt.legend()
plt.grid(True)

