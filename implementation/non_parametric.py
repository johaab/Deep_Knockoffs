### This script contains functions necessary to perform the Non-Parametric Tests and threshold the activations.

import numpy as np
import scipy.io
from implementation.utils import BETA_DIR
import implementation.load as load

def threshold(real, ci_array):
    if (real <= ci_array[0]):
        return 1
    elif (real >= ci_array[1]):
        return 1
    return 0


def uncorrected_test(val, alpha=.05):
    '''
    Perform an uncorrected region level non-paramatric test on data of 1 subject.
    :param val: np.array of dimension (num knockoffs, regions, paradigms). index 0 is the true value
    :param alpha: Two sided threshold, alpha is the test ex: alpha .05 is 95% level
    :return: np.array (region, thresholded beta) 1 for right side reject, -1 for left side reject, 0 accept
    '''
    print("Performing uncorrected non-parametric test...")
    region = val.shape[1]
    ko = val.shape[0]
    ci = [int(ko * alpha/2), int(ko * (1 - alpha/2))]  # confidence interval

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


def corrected_test(val, alpha=.05):
    '''
    Perform a single threshold image wise non-paramatric test on data of 1 subject.
    :param val: np.array of dimension (num knockoffs, regions, knock_betas). index 0 is the true value
    :param alpha: Two sided threshold, alpha is the test ex: alpha .05 is 95% level
    :return: np.array (region, thresholded beta) 1 for right side reject, -1 for left side reject, 0 accept
    '''
    print("Performing corrected non-parametric test...")
    paradigm = val.shape[2]
    regions = val.shape[1]
    ko = val.shape[0]
    ci = [int(ko * alpha/2), int(ko * (1 - alpha/2))] # confidence interval
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


def sl_test(selected_subjects=None, selected_task='all', save=False):
    '''
    :param selected_subjects: list subjects selected for group analysis
    :param selected_task: task selected for group analysis
    :return: uncorrected and corrected non-parametric testing of second-level empirical betas using surrogates
    '''

    if selected_subjects is None:
        raise ValueError('The group of subjects must be specified')

    tasks = ['MOTOR', 'GAMBLING', 'RELATIONAL', 'SOCIAL', 'WM', 'EMOTION', 'LANGUAGE']

    if selected_task != 'all':
        assert selected_task in tasks, "Task must be a value in - [EMOTION, GAMBLING, LANGUAGE, MOTOR, RELATIONAL, SOCIAL, WM] or 'all' "
        tasks = [selected_task]

    uncorrected_betas = []
    corrected_betas = []
    for task in tasks:
        empirical_sl_betas = scipy.io.loadmat(f'data/output/beta/empirical_sl_betas_t{task}.mat')['beta']
        surrogate_sl_betas = scipy.io.loadmat(f'data/output/beta/DeepKO_sl_betas_t{task}.mat')['beta']

        ### Uncorrected non-parametric test
        uncorrected_threshold = sl_uncorrected_test(empirical_sl_betas, surrogate_sl_betas)
        uncorrected_betas.append(get_corrected_betas(uncorrected_threshold, empirical_sl_betas))

        ### Corrected non-parametric test
        corrected_threshold = sl_corrected_test(empirical_sl_betas, surrogate_sl_betas)
        corrected_betas.append(get_corrected_betas(corrected_threshold, empirical_sl_betas))

        if save:
            load.save_mat(uncorrected_betas, BETA_DIR, 'DeepKO_uncorrected_sl_betas', task)
            load.save_mat(corrected_betas, BETA_DIR, 'DeepKO_corrected_sl_betas', task)


    return uncorrected_betas, corrected_betas



def sl_uncorrected_test(empirical, surrogate, alpha=.05):
    '''
    Perform an uncorrected non-parametric test on second-level betas (i.e. for group analysis)
    :param empirical: empirical second-level betas to test
    :param surrogate: surrogate second-level betas used to build the null distribution
    :param alpha: significance level
    :return: list of 0 and 1, 0 for unsignificant (de)activation, 1 for significant (de)activation at specific location
    '''
    print("Performing uncorrected non-parametric test on second-level betas...")

    regions = empirical.shape[0]
    ko = surrogate.shape[0] # we have n surrogates for each subject of the group
    ci = [int(ko * alpha/2), int(ko * (1 - alpha/2))]  # confidence interval

    uncorrected_threshold = []
    # Loop over brain regions and at each region for each beta perform a hypothesis test
    for region in range(regions):

        real = empirical[region, :]
        sort_reg = np.sort(surrogate[:, region, :], axis=0)
        ci_array = sort_reg[ci, :]
        ind = []
        # For each beta of the real subject: compare against confidence interval and accept or reject H0
        for b in range(len(real)):
            ind.append(threshold(real[b], ci_array[:, b]))
        uncorrected_threshold.append(ind)
    return np.array(uncorrected_threshold)

def sl_corrected_test(empirical, surrogate, alpha=.05):
    '''
    Perform a corrected non-parametric test on second-level betas (i.e. for group analysis)
    :param empirical: empirical second-level betas to test
    :param surrogate: surrogate second-level betas used to build the null distribution
    :param alpha: significance level
    :return: list of 0 and 1, 0 for unsignificant (de)activation, 1 for significant (de)activation at specific location
    '''
    print("Performing corrected non-parametric test on second-level betas...")

    regions = empirical.shape[0]
    conditions = empirical.shape[1]
    ko = surrogate.shape[0]  # we have n surrogates for each subject of the group
    ci = [int(ko * alpha/2), int(ko * (1 - alpha/2))]  # confidence interval
    corrected_threshold = []

    # Loop over each task condition and calculate maximal statistic for each beta over all brain regions.
    for condition in range(conditions):

        max_ = np.amax(surrogate[:, :, condition], axis=1)  # Takes the maximum per image (k+1 if k is num of knockoffs)
        min_ = np.amin(surrogate[:, :, condition], axis=1)  # Takes the minimum per image (k+1 if k is num of knockoffs)
        image_beta_max = np.sort(max_)
        image_beta_min = np.sort(min_)
        ci_array = [image_beta_min[ci[0]], image_beta_max[ci[1]]]

        ind = []
        # Compare beta values against maximal thresholded values and accept or reject null hypothesis.
        for beta in empirical[:, condition]:
            ind.append(threshold(beta, ci_array))
        corrected_threshold.append(ind)

    return np.array(corrected_threshold).T
