### This script contains functions necessary to perform the Non-Parametric Tests and threshold the activations.

import numpy as np


def threshold(real, ci_array):
    if (real <= ci_array[0]):
        return 1
    elif (real >= ci_array[1]):
        return 1
    return 0


def uncorrected_test(val, alpha=.025):
    '''
    Perform an uncorrected voxel level non-paramatric test on data of 1 subject.
    :param val: np.array of dimension (num knockoffs, regions, paradigms). index 0 is the true value
    :param alpha: Two sided threshold, 2*alpha is the test ex: alpha .05 is 90% level
    :return: np.array (region, thresholded beta) 1 for right side reject, -1 for left side reject, 0 accept
    '''
    print("Performing uncorrected non-parametric test...")
    region = val.shape[1]
    n = val.shape[0]
    ci = [int(n * alpha), int(n * (1 - alpha))]  # confidence interval

    uncorrected_threshold = []
    # Loop over brain regions and at each region for each beta perform a hypothesis test
    for i in range(region):
        reg = val[:, i, :]
        real = reg[0, :]
        sort_reg = np.sort(reg, axis=0)
        ci_array = sort_reg[ci, :]
        ind = []
        # For each beta of the real subject: compare against confidence interval and accept or reject H0
        for b in range(len(real)):
            ind.append(threshold(real[b], ci_array[:, b]))
        uncorrected_threshold.append(ind)
    return np.array(uncorrected_threshold)


def corrected_test(val, alpha=.025):
    '''
    Perform a single threshold image wise non-paramatric test on data of 1 subject.
    :param val: np.array of dimension (num knockoffs, regions, knock_betas). index 0 is the true value
    :param alpha: Two sided threshold, 2*alpha is the test ex: alpha .05 is 90% level
    :return: np.array (region, thresholded beta) 1 for right side reject, -1 for left side reject, 0 accept
    '''
    print("Performing corrected non-parametric test...")
    paradigm = val.shape[2]
    regions = val.shape[1]
    ko = val.shape[0]
    ci = [int(ko * alpha), int(ko * (1 - alpha))] # confidence interval
    corrected_threshold = []

    # Loop over each paradigm and calculate maximal statistic for each beta over all brain regions.
    for i in range(paradigm):
        brain = val[:, :, i]
        real = brain[0, :]
        max_ = np.amax(brain, axis=1) # Takes the maximum per image (k+1 if k is num of knockoffs)
        min_ = np.amin(brain, axis=1) # Takes the minimum per image (k+1 if k is num of knockoffs)
        image_beta_max = np.sort(max_)
        image_beta_min = np.sort(min_)
        ci_array = [image_beta_min[ci[0]], image_beta_max[ci[1]]]

        ind = []
        # Compare beta values against maximal thresholded values and accept or reject null hypothesis.
        for reg in real:
            ind.append(threshold(reg, ci_array))
        corrected_threshold.append(ind)
    return np.array(corrected_threshold).T


def get_corrected_betas(corrected, betas):
    '''
    :param corrected: thresholded activations
    :param betas: beta values from the GLM
    :return: thresholded_betas: thresholded knock_betas after non-parametric tests
    '''
    thresholded_betas = corrected * betas
    return thresholded_betas