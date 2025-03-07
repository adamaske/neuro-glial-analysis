import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spc

def gamma(x, alpha, beta):
    return (beta**alpha / spc.gamma(alpha)) * x**(alpha) * np.exp(-beta * x)

def chrf(t, a1, a2, b1, b2, c):
    """ Computes a double gamma canonical hemodynamic response function."""
    hrf = gamma(t, a1, b2) - (c* gamma(t, a2,b2))
    return hrf

def test_thetas():
    thetas = [[6, 16, 1, 1, 1/6],
              [5, 17, 2, 1, 1/10],
              [4, 18, 1, 3, 1/12],
              ]

    colors = [ "red", "green", "blue", "orange"]
    duration = 30
    sampling_frequency = 5.1
    time = np.linspace(0, duration, int(duration*sampling_frequency))
    for i, t in enumerate(thetas):
        a1 = t[0]
        a2 = t[1]
        b1 = t[2]
        b2 = t[3]
        c = t[4]

        hrf = chrf(time, a1, a2, b1, b2, c)
        print(f"cHRF : a1={a1}, a2={a2}, b1={b1}, b2={b2}, c={c}")

        label = f"a1={a1}, a2={a2}, b1={b1}, b2={b2}, c={c}" #create label
        plt.plot(time, hrf, color=colors[i], label=label)


    plt.title(f'Double Gamma HRF', fontsize=20)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)
    plt.legend(fontsize=20) # Adjust fontsize here
    plt.grid(True)
    plt.show()

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

