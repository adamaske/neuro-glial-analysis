import numpy as np
import matplotlib.pyplot as plt

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

def test_beta_value_calculation():
    #Parameters
    duration = 15  # seconds
    sampling_rate = 5  # Hz
    num_samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, num_samples)
    mean = 6
    std_dev = 2
    # Gaussian time series
    gaussian_time_series = np.exp(-0.5 * ((time - mean) / std_dev)**2)

    # Boxcar function (same as before)
    boxcar_function = np.zeros(num_samples)
    start_index = int(5 * sampling_rate)
    end_index = int(10 * sampling_rate)
    boxcar_function[start_index:end_index] = 1

    from hrf import double_gamma_chrf
    hrf = double_gamma_chrf(time, 6, 16, 1, 1, 1/6)
    hrf_beta = glm(y=gaussian_time_series, x=hrf)
    boxcar_beta = glm(y=gaussian_time_series, x=boxcar_function)

    print(f"hrf_beta : ", hrf_beta)
    print(f"boxcar_beta : ", boxcar_beta)

    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time, gaussian_time_series)
    plt.title("Gaussian Time Series (5 Hz)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, boxcar_function)
    plt.title("Boxcar Function (5-10 seconds)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, hrf)
    plt.title("Doble Gamma cHRF")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.show()
    
def test_sliding_window_glm():
    
    
    from hrf import double_gamma_chrf
    
    
    def sliding_window_glm(y, x,  num_samples, sample_rate, window_size, step_size,):
        # Here there are two alternatives, do we want to apply the same X across each window, or should there also
        # be a window into the x

        window_size = int(window_size * sample_rate)  
        step_size = int(step_size * sample_rate) 
        beta_values = np.zeros(num_samples)
        
        for i in range(0, num_samples- window_size+(window_size // 2), step_size):
            window_start = i
            window_end = i + window_size
    
            window_y = y[window_start:window_end]
            hrf = double_gamma_chrf(np.linspace(0, window_y.shape[0], window_y.shape[0]), 6, 16, 1, 1, 1/6)
            
            beta = glm(window_y, hrf)
                
            center_index = window_start + (window_size // 2)
            beta_values[center_index] = beta #add beta value to the correct index
            
            
            #plt.subplot(2, 1, 1)
            #
            #plt.title(f"Signal : {window_start/sample_rate}-{window_end/sample_rate}")
            #plt.plot(window_y)
            #plt.subplot(2, 1, 2)
            #plt.title(f"HRF a1=6, a2=16, b1=b2=1, c=1/6")
            #plt.plot(hrf)
            #
            #plt.show()
            #print("window_y : ", window_y.shape)
            #print("hrf :", hrf.shape)
            
        return beta_values
    
    #Parameters
    duration = 15  # seconds
    sampling_rate = 5  # Hz
    num_samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, num_samples)
    mean = 10
    std_dev = 2
    # Gaussian time series
    gaussian_time_series = np.exp(-0.5 * ((time - mean) / std_dev)**2)

    # Boxcar function (same as before)
    boxcar_function = np.zeros(num_samples)
    start_index = int(5 * sampling_rate)
    end_index = int(10 * sampling_rate)
    boxcar_function[start_index:end_index] = 1
    
    # HRF
    hrf = double_gamma_chrf(time, 6, 16, 1, 1, 1/6)
    
    betas = sliding_window_glm(gaussian_time_series, hrf, num_samples, sampling_rate, window_size=3, step_size=1)   
             
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time, gaussian_time_series)
    plt.title("Gaussian Time Series (5 Hz)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, hrf)
    plt.title("Doble Gamma cHRF")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time, betas)
    plt.title("Sliding Window Beta Values")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    test_beta_value_calculation()
    test_sliding_window_glm()