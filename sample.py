import numpy as np
from scipy import stats
import random
roi_1_hbo_chrf_betas = np.array([0.67, 0.55, 0.82, 0.67, 0.9, 0.8, 0.5, 0.6, 0.9])
roi_2_hbo_chrf_betas = np.array([0.5, 0.4, 0.6, 0.6, 0.5, 0.9, 0.55, 0.7, 0.56])

# Calculate the paired differences
paired_differences = roi_1_hbo_chrf_betas - roi_2_hbo_chrf_betas

# Calculate the mean of the paired differences
mean_difference = np.mean(paired_differences)

# Calculate the standard deviation of the paired differences
std_difference = np.std(paired_differences, ddof=1)  # ddof=1 for sample standard deviation

# Calculate Cohen's d
cohens_d = mean_difference / std_difference

print(f"Mean difference: {mean_difference:.3f}")
print(f"Standard deviation of differences: {std_difference:.3f}")
print(f"Cohen's d: {cohens_d:.3f}")
exit()
def two_roi_p_value(roi1, roi2):
    """
    Args:
        Beta values for each subject
    
    """
    t_stat, p_val = stats.ttest_rel(roi1, roi2)
    return t_stat, p_val

def check_signficance(alpha, p_value):
    if p_value < alpha:
        print("The difference between the two ROIs is statistically significant.")
    else:
        print("There is no statistically significant difference between the two ROIs.")


import analysis_info 
info = analysis_info.get_info()
rois = info["regions_of_interest"]
subjects = info["subject_ids"]
events = info["events"]

data = [] # Load from somewhere

# Lateralization Index
for roi in rois:
    for event in events: # i.e Pronation, Supination, Neutral
        lh_betas = [] # Left hemisphere beta values for every subject
        rh_betas = [] # Right hemisphere

        for subject in subjects:
            lh_betas.append(random.randint(0, 100))
            rh_betas.append(random.randint(0, 100))
        
        t_stat, p_val = two_roi_p_value(lh_betas, rh_betas)

        print(f"Lateralization Index : {roi} [ {events[event]} ]")
        print(f"Subjects : {len(subjects)}")
        print(f"Left Hemispehre : ", lh_betas)
        print(f"Right Hemispehre : ", rh_betas)

        print("Paired t-test results:")
        print(f"t-statistic: {t_stat}")
        print(f"p-value: {p_val}")
        print()
exit()
# For every ROI
# For every subject
# Get the 

# Each element is the beta value calculated for a subject's roi 
# The elements in each are from the same subject i.e roi1[0] is the same subject as roi2[0], 
roi_1_hbo_chrf_betas = [0.67, 0.55, 0.82, 0.67, 0.9, 0.8, 0.5, 0.6, 0.9]
roi_2_hbo_chrf_betas = [0.2, 0.1, 0.3, 0.4, 0.1, 0.2, 0.5, 0.3, 0.2]

print("std : ", np.std(roi_1_hbo_chrf_betas))
print("std : ", np.std(roi_2_hbo_chrf_betas))

# Is there a significant difference between the two ROIs: S1 and S2?
# Do a paired t-test since the data is paired (same subjects)

t_stat, p_val = two_roi_p_value(roi_1_hbo_chrf_betas, roi_2_hbo_chrf_betas)

print("Paired t-test results:")
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_val}")

# Interpretation of the p-value
alpha = 0.05  # Significance level
if p_val < alpha:
    print("The difference between the two ROIs is statistically significant.")
else:
    print("There is no statistically significant difference between the two ROIs.")

#calculate cohens d effect size for paired t test.
diff = np.array(roi_1_hbo_chrf_betas) - np.array(roi_2_hbo_chrf_betas)
cohens_d = np.mean(diff) / np.std(diff, ddof=1)
print(f"Cohen's d: {cohens_d}")