import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spc

def gamma(x, alpha, beta):
    return (beta**alpha / spc.gamma(alpha)) * x**(alpha) * np.exp(-beta * x)

def double_gamma_chrf(t, a1, a2, b1, b2, c):
    """ Computes a double gamma canonical hemodynamic response function."""
    hrf = gamma(t, a1, b1) - (c* gamma(t, a2, b2))
    return hrf

def test_thetas():
    thetas = [[6, 16, 1, 1, 1/6],
              [5.2, 10.8, 1, 1, 1/6],
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

        hrf = double_gamma_chrf(time, a1, a2, b1, b2, c)
        print(f"cHRF : a1={a1}, a2={a2}, b1={b1}, b2={b2}, c={c}")

        label = f"a1={a1}, a2={a2}, b1={b1}, b2={b2}, c={c}" #create label
        plt.plot(time, hrf, color=colors[i], label=label)


    plt.title(f'Double Gamma HRF', fontsize=20)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)
    plt.legend(fontsize=20) # Adjust fontsize here
    plt.grid(True)
    plt.show()


#test_thetas()