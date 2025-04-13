import numpy as np
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

# Example: assume you have N subjects and connectivity matrices of shape (channels x channels)
N_subjects = 20      # change this to your actual number
N_channels = 32      # change this to match your EEG or fNIRS setup

# Mock data: replace these with your real matrices
# S_matrices and P_matrices should each be a list of NumPy arrays
S_matrices = [np.random.rand(N_channels, N_channels) for _ in range(N_subjects)]
P_matrices = [np.random.rand(N_channels, N_channels) for _ in range(N_subjects)]

# Step 1: Calculate Difference Matrices for each subject
D_matrices = [S - P for S, P in zip(S_matrices, P_matrices)]

# Step 2: Stack differences into a 3D array for easier testing
D_stack = np.stack(D_matrices)  # shape: (subjects, channels, channels)

# Step 3: Perform one-sample t-test across subjects for each channel pair
t_values = np.zeros((N_channels, N_channels))
p_values = np.zeros((N_channels, N_channels))

for i in range(N_channels):
    for j in range(N_channels):
        # One-sample t-test against zero
        t_stat, p_val = ttest_1samp(D_stack[:, i, j], 0.0)
        t_values[i, j] = t_stat
        p_values[i, j] = p_val

# Step 4: Multiple Comparisons Correction (FDR)
p_values_flat = p_values.flatten()
reject, pvals_corrected, _, _ = multipletests(p_values_flat, alpha=0.05, method='fdr_bh')

# Reshape corrected p-values back to matrix form
pvals_corrected_matrix = pvals_corrected.reshape(N_channels, N_channels)
reject_matrix = reject.reshape(N_channels, N_channels)

# Results:
print("Corrected p-values matrix:\n", pvals_corrected_matrix)
print("Significant connections after FDR:\n", reject_matrix)

