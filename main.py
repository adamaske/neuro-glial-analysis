import scipy.stats as stats
import numpy as np

# Example data (replace with your actual beta values)
case1_betas = np.array([1.2, 1.5, 1.8, 2.0, 1.3])  # Beta values for case 1
case2_betas = np.array([1.0, 1.3, 1.6, 1.9, 1.1])  # Beta values for case 2

# Perform the paired t-test
t_statistic, p_value = stats.ttest_rel(case1_betas, case2_betas)

# Print the results
print("T-statistic:", t_statistic)
print("P-value:", p_value)

# Interpret the p-value (assuming alpha = 0.05)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference.")

exit()
def li_f1(a, b):
    li = (a - b) / (a + b)
    return li

def li_f2(a,b):
    li = (a - b) / (abs(a) + abs(b))
    return li

def li_f3(a,b):
    li = (abs(a) - abs(b)) / (abs(a) + abs(b))
    return li

def test_vals(a, b):
    print(f" LI : {a} vs {b}")
    f1 = li_f1(a, b)
    f2 = li_f2(a, b)
    f3 = li_f3(a, b)

    print("f1 : ", f1) 
    print("f2 : ", f2) 
    print("f3 : ", f3) 
    
    return f1, f2, f3


test_vals(1.2, 0.1)
test_vals(1.2, -0.1)
test_vals(-1.2, 0.1)
test_vals(-1.2, -0.1)

