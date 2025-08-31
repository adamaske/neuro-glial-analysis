import numpy as np
import matplotlib.pyplot as plt
from hrf import double_gamma_chrf
from neuropipeline.fnirs import fNIRS

tmin = 0
tmax = 20

fnirs = fNIRS()
# Load fNIRS data

# split HbO HbR


# For subject 1
# For trial 1
# epochs = 
# epoch_avg = []
# for onset in fnirs.feature_onsets:
#       epoch = hbo_data[:, (onset + tmin):(onset+tmax)]
#       epochs.append(epoch)
# block_average = np.mean(epochs)
# t = Shape of epoch
# hrf = double_gamma_chrf(t, 6, 16, 1, 1, 1/6) # As noted in Lindquist 2009
# beta = b_value(epoch, hrf)
#
# if fnirs.feature_descriptions[0] == "3" : Pronation
# elif fnirs.feature_descriptions[0] == "4" : Supination
#       save the block average figure and beta value
# -> figname = "sub_" + i + "_trial_" + j + "_pronation_" + ch_name + "_" + beta ".png" 
# plt.savefig()
# Block 1, 2, 3

t = np.linspace(0, 20, 100)
hrf = double_gamma_chrf(t, 6, 16, 1, 1, 1/6) # As noted in Lindquist 2009

tmin = 0
tmax = 20

#Parameters
duration = 15  # seconds
sampling_rate = 5  # Hz
num_samples = int(duration * sampling_rate)
time = np.linspace(0, duration, num_samples)
mean = 7
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

def glm(y, x):
    """
    Y = BX + e\n
    B = (X^T*X)^-1 * X^T * Y\n
    Args:
        y : One dimensional time series - signal\n
        x : One dimensional time series - 
    Returns:
        B : Beta value (sensitivty of Y to X)
    """
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    
    x_transpose_x = np.matmul(X.T, X)
    try:
        x_transpose_x_inverse = np.linalg.inv(x_transpose_x)
    except np.linalg.LinAlgError:
        x_transpose_x_inverse = 0 

    x_transpose_y = np.matmul(X.T, Y)
    B = np.matmul(x_transpose_x_inverse, x_transpose_y)[0, 0] #extract scalar beta
    
    return B

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

<<<<<<< Updated upstream
plt.subplot(4, 2, 3)
=======
plt.subplot(4, 1, 3)
plt.plot(time, convolution_result)
plt.title("convolution_result (5-10 seconds)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 4)
>>>>>>> Stashed changes
plt.plot(time, beta_values, label="Sliding Window Beta Values")
plt.title("Sliding Window Beta Values")
plt.xlabel("Time (seconds)")
plt.ylabel("Beta Value")
plt.ylim((0, 1))
plt.legend()
plt.grid(True)

<<<<<<< Updated upstream
plt.show()
=======
plt.show()

>>>>>>> Stashed changes
