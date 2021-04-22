import matplotlib.pyplot as plt
import scipy.io
import seaborn as sns
from os.path import join
import numpy as np

from implementation.utils import BETA_DIR

#####################################################################
## Intitialisation: a task and a subject
#####################################################################
# select a task in ['MOTOR', 'GAMBLING', 'RELATIONAL', 'SOCIAL', 'WM', 'EMOTION', 'LANGUAGE']
task = 'MOTOR'
subject = 1
alpha = .05

plot_uncorrected = True
plot_corrected = True

# Load the betas of the empirical data
beta_path = join(BETA_DIR, f"GLM_betas_t{task}")
true_betas = scipy.io.loadmat(beta_path)['beta'][subject, :, :]

# Load the surrogate betas
deepko_betas = scipy.io.loadmat(f'data/output/beta/DeepKO_betas_t{task}_s{subject}.mat')['data']

conditions = np.arange(true_betas.shape[1])

if plot_uncorrected:
    num_bins = 20
    region = 100

    ko = deepko_betas.shape[0]
    ci = [int(ko * alpha / 2), int(ko * (1 - alpha / 2))]

    avg_region_true_betas = np.average(true_betas, axis=0)
    avg_cond_true_betas = np.average(true_betas, axis=1)

    for condition in conditions:
        plot_deepko_betas = plt.figure()
        ci_array = np.sort(deepko_betas, axis=0)[ci, region, condition]
        sns.histplot(deepko_betas[1:, region, condition], bins=num_bins, kde=True, edgecolor='black',
                     label='Surrogates')
        plt.axvline(true_betas[region, condition], color='orange', label=f'Empirical')
        plt.axvline(ci_array[0], color='red', linestyle='--')
        plt.axvline(ci_array[1], color='red', linestyle='--', label='Threshold')
        plt.legend()
        # plt.title(f'Condition {condition}')
        plt.xlabel('Value')
        plt.ylabel('Count')

        plot_deepko_betas.set_figwidth(6)
        plot_deepko_betas.set_figheight(4)

        plt.savefig(f'uncorrected_surrogate_distribution_t{task}_s{subject}_region{region}_condition{condition}')

if plot_corrected:
        regions = true_betas.shape[0]
        ko = deepko_betas.shape[0]
        ci = [int(ko * alpha / 2), int(ko * (1 - alpha / 2))]  # confidence interval
        corrected_threshold = []

        # Loop over each task condition and calculate maximal statistic for each beta over all brain regions.
        for condition in conditions:

            brain = deepko_betas[:, :, condition]
            real = true_betas[:, condition]
            max_ = np.amax(brain, axis=1)  # Takes the maximum per image (k+1 if k is num of knockoffs)
            min_ = np.amin(brain, axis=1)  # Takes the minimum per image (k+1 if k is num of knockoffs)
            image_beta_max = np.sort(max_)
            image_beta_min = np.sort(min_)
            ci_array = [image_beta_min[ci[0]], image_beta_max[ci[1]]]

            plot_corr_deepko_betas = plt.figure()
            num_bins = 20

            #plt.subplot(211) # MIN
            # Plot surrogate distribution
            sns.histplot(min_[1:], bins=num_bins, kde=True, edgecolor='black', label='$T^{min}$')
            # Plot empirical distribution
            sns.histplot(real[real < 0], bins=2 * num_bins, kde=True, alpha=.3, color='orange', label='Empirical < 0')
            # Highlight empirical data
            # plt.axvline(x=min_[0], color='orange', linewidth=5, label="Min empirical data")
            # Mark test threshold
            plt.axvline(x=ci_array[0], color='r', linestyle='--', label="Threshold")
            plt.xlabel('Value')
            plt.ylabel('Count')
            # plt.xlim([-1, 0])
            #plt.title(f"Minimum statistic (alpha={alpha / 2})")
            #plt.legend(bbox_to_anchor=(-0.03, 0.3))
            plt.legend(loc='upper left')
            plot_corr_deepko_betas.set_figwidth(6)
            plot_corr_deepko_betas.set_figheight(4)
            plt.savefig(f'corrected_surrogate_distribution_t{task}_s{subject}_condition{condition}_left')

            plot_corr_deepko_betas = plt.figure()
            #plt.subplot(212) # MAX
            # Plot surrogate distribution
            sns.histplot(max_[1:], bins=num_bins, kde=True, edgecolor='black', label='$T^{max}$')
            # Plot empirical distribution
            sns.histplot(real[real > 0], bins=2 * num_bins, kde=True, alpha=.3, color='orange', label='Empirical > 0')
            # Highlight empirical data
            # plt.axvline(x=max_[0], color='orange', linewidth=5, label="Max empirical data")
            # Mark test threshold
            plt.axvline(x=ci_array[1], color='r', linestyle='--', label="Threshold")
            plt.xlabel('Value')
            plt.ylabel('Count')
            # plt.xlim([0, 1])
            #plt.title(f"Maximum statistic (alpha={alpha / 2})")
            #plt.legend(bbox_to_anchor=(-0.03, 0.3))
            plt.legend(loc='upper right')

            plot_corr_deepko_betas.set_figwidth(6)
            plot_corr_deepko_betas.set_figheight(4)
            plt.savefig(f'corrected_surrogate_distribution_t{task}_s{subject}_condition{condition}_right')