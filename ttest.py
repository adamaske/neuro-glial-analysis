import scipy.stats as stats
import numpy as np

data = np.array([0.78, 0.62, 0.3, 0.5, 0.1]) # example data.
baseline_value = 0.5

t_statistic, p_value = stats.ttest_1samp(data, baseline_value)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")