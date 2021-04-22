### Group analysis
# Plan:
# - perform the individual analysis with the surrogate timecourses
# - consider the surrogate first-level betas obtained from the individual analysis to enter the second level analysis,
#   you would obtain then surrogate betas’, without redoing the NN training, but just producing the initial time course surrogates.
# - go ahead and test second level empirical betas' against surrogate betas’

import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt

from implementation import glm, knockoff_class
from implementation.load import load_fmri
from implementation import non_parametric

run_glm = False
generate_fl_betas = False
plot_betas = True

#####################################################################
## Initialisation: a task and some subjects
#####################################################################

# select a task in ['MOTOR', 'GAMBLING', 'RELATIONAL', 'SOCIAL', 'WM', 'EMOTION', 'LANGUAGE'] or 'all'
task = 'MOTOR'
# select subjects to form a group
group = np.arange(2)

# loading all the data
fmri_data = load_fmri(task=task)
n_subjects = fmri_data.shape[0]
n_regions = fmri_data.shape[1]
n_time = fmri_data.shape[2]
n_conditions = 5

if np.amin(group) < 0 or np.amax(group) >= n_subjects:
    raise ValueError('The group was mis-specified')

assert n_regions == 379, 'error regions dont match'

#####################################################################
## GLM
#####################################################################
if run_glm:

    glm.run(selected_task=task)

else:
    # check if the required files already exist to load them afterward
    print('Checking GLM file existence...')

    if not os.path.isfile(f'data/output/beta/GLM_betas_t{task}.mat'):
        ValueError("File containing first-level betas not found, you must run GLM")

    if not os.path.isfile(f'data/output/beta/GLM_uncorrected_betas_t{task}.mat'):
        ValueError("File containing uncorrected first-level betas not found, you must run GLM")

    if not os.path.isfile(f'data/output/beta/GLM_corrected_betas_t{task}.mat'):
        ValueError("File containing uncorrected first-level betas not found, you must run GLM")

    print('Done!')

#####################################################################
## DKO
#####################################################################

fmri_data = fmri_data[group]
n_subjects = fmri_data.shape[0]
n_surrogates = 100
surrogate_fl_betas = np.empty((n_subjects, n_surrogates+1, n_regions, n_conditions))


for subject in group:
    if generate_fl_betas:
        print(f'==============================\n Subject {subject+1}/{len(group)} \n==============================')
        # Initialisation
        deepko = knockoff_class.DeepKnockOff(task, int(subject))
        # Pre-process the empirical fMRI data (single-linkage clustering to calculate approximate covariance matrix)
        deepko.pre_process(max_corr=.4, save=True)

        # Training the machine to build high-order knockoffs. The parameters can be changed at params.py
        _ = deepko.fit()

        # Built the knockoffs
        data_deepko = deepko.transform(iters=n_surrogates, save=True)

        # Run GLM on knockoffs to get surrogate betas
        surrogate_fl_betas[subject] = deepko.statistic(data_deepko, save=True)

    else:
        print(f'Checking first-level beta file existence for task {task} and subject {subject}...')

        if not os.path.isfile(f'data/output/beta/DeepKO_betas_t{task}_s{subject}.mat'):
            ValueError("File containing first-level betas not found, you must train the Deep Knockoff machine and run first-level GLM on surrogate fMRI timecourses")

        print('Done!')
        #surrogate_fl_betas[subject] = scipy.io.loadmat(f'data/output/beta/DeepKO_betas_t{task}_s{subject}.mat')['data']


#####################################################################
## Second-level analysis
#####################################################################
# Run second-level GLM on empirical first-level betas
glm.run_second_level(selected_subjects=group, selected_task=task)

# Non-parametric testing
nonpara_uncorrected_betas, nonpara_corrected_betas, = non_parametric.sl_test(selected_subjects=group, selected_task=task, save=True)

#####################################################################
## Plot
#####################################################################
print("Plotting...")

empirical_sl_betas = scipy.io.loadmat(f'data/output/beta/empirical_sl_betas_t{task}.mat')['beta']
empirical_uncorrected_sl_betas = scipy.io.loadmat(f'data/output/beta/empirical_uncorrected_sl_betas_t{task}.mat')['beta']
empirical_corrected_sl_betas = scipy.io.loadmat(f'data/output/beta/empirical_corrected_sl_betas_t{task}.mat')['beta']
dko_uncorrected_sl_betas = scipy.io.loadmat(f'data/output/beta/DeepKO_uncorrected_sl_betas_t{task}.mat')['beta']
dko_corrected_sl_betas = scipy.io.loadmat(f'data/output/beta/DeepKO_corrected_sl_betas_t{task}.mat')['beta']

if plot_betas:
    #show_condition = 0
    show_regions = np.arange(50)
    for show_condition in range(n_conditions):

        betas_selection = plt.figure()

        # Parametric testing
        plt.subplot(211)

        plt.bar(show_regions, empirical_sl_betas[show_regions, show_condition], width=.2, color='red', label="Second-level empirical betas")
        plt.bar(show_regions + .2, empirical_uncorrected_sl_betas[show_regions, show_condition], width=.2, color='blue',
                label="Uncorrected")
        plt.bar(show_regions + .4, empirical_corrected_sl_betas[show_regions, show_condition], width=.2, color='green',
                label="Corrected (using Bonferroni)")
        plt.xlabel('Region')
        plt.ylabel('Value')
        plt.title("Parametric testing")
        plt.legend(loc='lower right')

        # Non-parametric testing
        plt.subplot(212)

        plt.bar(show_regions, empirical_sl_betas[show_regions, show_condition], width=.2, color='red', label="Second-level empirical betas")
        plt.bar(show_regions + .2, dko_uncorrected_sl_betas[0, show_regions, show_condition], width=.2, color='blue',
                label="Uncorrected")
        plt.bar(show_regions + .4, dko_corrected_sl_betas[0, show_regions, show_condition], width=.2, color='green',
                label="Corrected (using Min/Max)")
        plt.xlabel('Region')
        plt.ylabel('Value')
        plt.title("Non-parametric testing")
        plt.legend(loc='lower right')

        betas_selection.suptitle(
            f"Investigation of Betas selection for group analysis (task: {task}, subjects: {group[0]}-{group[-1]}, condition: {show_condition})", fontsize=20)

        betas_selection.set_figwidth(15)
        betas_selection.set_figheight(10)
        betas_selection.savefig(f'betas_selection_task{task}_condition{show_condition}_subjects{group[0]}-{group[-1]}')

print('Done!')