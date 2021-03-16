import scipy.io
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math

def threshold(real, ci_array):
    if (real <= ci_array[0]):
        return 1
    elif (real >= ci_array[1]):
        return 1
    return 0

#
# Need to be placed at the root of the repository in order to run
#

### Define significance level and condition (paradigm)
# Significance level
alpha = 0.05
# Condition (paradigm) selected
condition = 0
# Task
task = 'MOTOR'
# Subject
subject = 1
# Surrogates
N = 100

### Load data
## Data.shape = (N+1,379,5)
# N+1 = 1 real fMRI + N Knockoffs surrogates
# 379 voxels
# 5 paradigms
# 1 subject
val = scipy.io.loadmat('data/output/beta/DeepKO_KObetas_tMOTOR_s1.mat')['data']
ko = val.shape[0]
regions = val.shape[1]
paradigm = val.shape[2]
ci = [int(ko * alpha / 2), int(ko * (1 - alpha / 2))]  # confidence interval
corrected_threshold = []

assert ko == N+1 , "Loaded data and surrogate number specified don't match"

brain = val[:, :, condition]
real = brain[0, :]
max_ = np.amax(brain, axis=1)  # Takes the maximum per image (k+1 if k is num of knockoffs)
min_ = np.amin(brain, axis=1)  # Takes the minimum per image (k+1 if k is num of knockoffs)
image_beta_max = np.sort(max_)
image_beta_min = np.sort(min_)
ci_array = [image_beta_min[ci[0]], image_beta_max[ci[1]]]

ind = []
# Compare beta values against maximal thresholded values and accept or reject null hypothesis.
for reg in real:
    ind.append(threshold(reg, ci_array))
corrected_threshold.append(ind)

plot = plt.figure()
num_bins = 19

# Left tail
plt.subplot(211)
# Plot surrogate distribution
sns.histplot(min_[1:], bins=num_bins, kde=True, edgecolor='black', label='Min surrogate')
# Plot empirical distribution
sns.histplot(real[real < 0], bins=2*num_bins, kde=True, alpha=.3, color='orange', label='Empirical < 0')
# Highlight empirical data
plt.axvline(x=min_[0], color='orange', linewidth=5, label="Min empirical data")
# Mark test threshold
plt.axvline(x=ci_array[0], color='r', linestyle='--', label="Threshold")
plt.xlabel('$T_{min}$')
plt.ylabel('Count')
#plt.xlim([-1, 0])
plt.title(f"Minimum statistic (alpha={alpha/2})")
plt.legend(bbox_to_anchor=(-0.03, 0.3))

# Right tail
plt.subplot(212)
# Plot surrogate distribution
sns.histplot(max_[1:], bins=num_bins, kde=True, edgecolor='black', label='Max surrogate')
# Plot empirical distribution
sns.histplot(real[real > 0], bins=2*num_bins, kde=True, alpha=.3, color='orange', label='Empirical > 0')
# Highlight empirical data
plt.axvline(x=max_[0], color='orange', linewidth=5, label="Max empirical data")
# Mark test threshold
plt.axvline(x=ci_array[1], color='r', linestyle='--', label="Threshold")
plt.xlabel('$T_{max}$')
plt.ylabel('Count')
#plt.xlim([0, 1])
plt.title(f"Maximum statistic (alpha={alpha/2})")
plt.legend(bbox_to_anchor=(-0.03, 0.3))

plot.suptitle(f"Non-parametric testing (task:{task}, subject:{subject}, condition:{condition}, #surrogates:{N})", fontsize=20)
plot.set_figwidth(20)
plot.set_figheight(15)
#plt.show()
plt.savefig("surrogate_hist")