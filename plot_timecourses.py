import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import pandas as pd
import os
from os.path import join
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr

from implementation import glm, knockoff_class, params
import implementation.load as load
from implementation.load import load_pickle, load_fmri, load_hrf_function
from implementation.utils import KNOCK_DIR, BETA_DIR, PEARSONR_DIR, compare_diagnostics
from implementation.params import get_params

#####################################################################
## Intitialisation: a task and a subject
#####################################################################
# select a task in ['MOTOR', 'GAMBLING', 'RELATIONAL', 'SOCIAL', 'WM', 'EMOTION', 'LANGUAGE'] or 'all'
task = 'MOTOR'
# select a subject
subject = 1
# loading all the data
fmri_data = load_fmri(task=task)
fmri_data = fmri_data[subject, :, :]

run_glm = True
train = True
diagnostics = False
generate_betas = True
compute_correlation = False

plot_distribution = True
plot_timecourses = True
plot_correlation = False
plot_heatmap = False

#####################################################################
## Task paradigm
#####################################################################

task_paradigms = load.load_task_paradigms(task)
# do one hot encoding
task_paradigms_one_hot = load.separate_conditions(task_paradigms)

# do the convolution
hrf = load.load_hrf_function()
task_paradigms_conv = load.do_convolution(task_paradigms_one_hot, hrf)

task_paradigms_conv = task_paradigms_conv[subject, :, :]
task_paradigms_one_hot = task_paradigms_one_hot[subject, :, :]

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
        ValueError("File containing uncorrected second-level betas not found, you must run GLM")

    if not os.path.isfile(f'data/output/beta/GLM_corrected_betas_t{task}.mat'):
        ValueError("File containing uncorrected second-level betas not found, you must run GLM")

    #if not os.path.isfile(f'data/output/beta/rest_state_fMRI_{task}.mat'):
        #ValueError("File containing rest state fMRI not found, you must run GLM")

    print('Done!')

# Load the betas of the empirical data
beta_path = join(BETA_DIR, f"GLM_betas_t{task}")
true_betas = scipy.io.loadmat(beta_path)['beta'][subject, :, :]

#####################################################################
## DKO
#####################################################################

deepko = knockoff_class.DeepKnockOff(task, subject)
deepko.pre_process(max_corr=.4, save=True)

if train:
    # Training the machine to build high-order knockoffs. The parameters can be changed at params.py
    _ = deepko.fit()

    # Built the knockoffs
    data_deepko = deepko.transform(save=True)

else:
    # Loading previously trained machine
    _, x_train = load_pickle(KNOCK_DIR, f'DeepKO_sigma_hat_x_train_t{task}_s{subject}')
    groups, _ = load_pickle(KNOCK_DIR, f'DeepKO_groups_representatives_t{task}_s{subject}')
    params = load_pickle(KNOCK_DIR, f'DeepKO_tMOTOR_s{subject}_params')

    deepko.load_x(x_train)
    deepko.load_params(params)
    deepko.load_machine()

    # Built the knockoffs
    data_deepko = deepko.transform(groups=groups)


if generate_betas:
    # Run GLM on knockoffs to get surrogate betas
    deepko_betas = deepko.statistic(data_deepko, save=True)

#####################################################################
## Diagnostics
#####################################################################

if diagnostics:
    res_deepko = deepko.diagnostics()

#####################################################################
## Correlation
#####################################################################
epoch = get_params(0, 0, 0)['epochs'] # dummy arguments just to fetch the number of params

if compute_correlation:
    # Investigate correlation between individual surrogate and empirical fMRI timecourses
    pearsonr_corr = np.zeros([data_deepko.shape[0], data_deepko.shape[2], 2])

    for surrogate in range(data_deepko.shape[0]):
        for region in range(data_deepko.shape[2]):
            pearsonr_corr[surrogate, region, :] = pearsonr(fmri_data[region, :], data_deepko[surrogate, :, region])
            # can try with spearmanr (non-parametric) too
    load.save_mat(pearsonr_corr, PEARSONR_DIR, 'DeepKO_pearsonr', task+f'_s{subject}_epoch{epoch}')
else:
    pearsonr_corr = scipy.io.loadmat(f'data/output/pearsonr/DeepKO_pearsonr_t{task}_s{subject}_epoch{epoch}.mat')['beta']

#####################################################################
## Plot
#####################################################################

print("Plotting...")
groups, _ = load_pickle(KNOCK_DIR, f'DeepKO_groups_representatives_t{task}_s{subject}')

region = 0
cluster = groups[region]

if plot_distribution:

    num_bins = 20
    conditions = np.arange(true_betas.shape[1])

    alpha = .05
    ko = deepko_betas.shape[0]
    ci = [int(ko * alpha/2), int(ko * (1 - alpha/2))]

    avg_region_true_betas = np.average(true_betas, axis=0)
    avg_cond_true_betas = np.average(true_betas, axis=1)

    for condition in conditions:
        plot_deepko_betas = plt.figure()
        ci_array = np.sort(deepko_betas, axis=0)[ci, region, condition]
        sns.histplot(deepko_betas[1:, region, condition], bins=num_bins, kde=True, edgecolor='black', label='Surrogates')
        plt.axvline(true_betas[region, condition], color='orange', label=f'Empirical')
        plt.axvline(ci_array[0], color='red', linestyle='--')
        plt.axvline(ci_array[1], color='red', linestyle='--', label='Threshold')
        plt.legend()
        #plt.title(f'Condition {condition}')
        plt.xlabel('Value')
        plt.ylabel('Count')

        plot_deepko_betas.set_figwidth(20)
        plot_deepko_betas.set_figheight(10)

        plt.savefig(f'surrogate_distribution_t{task}_s{subject}_region{region}_condition{condition}')

    if False:
        for condition in conditions:
            #unique_deepko_betas = np.unique(deepko_betas[1:, region, condition])
            #print(f'unique_deepko_betas.shape condition {condition} : ', unique_deepko_betas.shape)
            ci_array = np.sort(deepko_betas, axis=0)[ci, region, condition]
            #ci_array = unique_deepko_betas[ci] # np.unique already sorts the array

            plt.subplot(2, 3, condition + 1)
            sns.histplot(deepko_betas[1:, region, condition], bins=num_bins, kde=True, edgecolor='black', label='Surrogates')
            #sns.histplot(unique_deepko_betas, bins=num_bins, kde=True, edgecolor='black', label='Surrogates')
            plt.axvline(true_betas[region, condition], color='orange', label=f'Empirical')
            plt.axvline(ci_array[0], color='red', linestyle='--')
            plt.axvline(ci_array[1], color='red', linestyle='--', label='Threshold')
            #plt.axvline(avg_region_true_betas[condition], color='green', linewidth=3, label='Empirical average over regions')
            #plt.axvline(avg_cond_true_betas[region], color='black', linewidth=3, label='Empirical average over condition')

            plt.legend()
            plt.title(f'Condition {condition}')
            plt.xlabel('Value')
            plt.ylabel('Count')

        plot_deepko_betas.set_figwidth(20)
        plot_deepko_betas.set_figheight(15)

        plot_deepko_betas.suptitle(
            f"Surrogate distribution for region {region} across task conditions (task:{task}, subject:{subject})",
            fontsize=20)

        plt.savefig(f'surrogate_distribution_t{task}_s{subject}_region{region}')

if plot_timecourses:
    plot_fmri = plt.figure()
    show_time = np.arange(280)

    if False:
        #cmap = plt.get_cmap('gnuplot')
        cmap = plt.get_cmap("gist_rainbow")
        colors = [cmap(i) for i in np.linspace(0, 1, 6)]

        plt.subplot(411)
        plt.plot(0, 0, color='black', linestyle='--', label='Paradigm')
        plt.plot(0, 0, color='black', linestyle='solid', label='Convo. with HRF')
        for condition in np.arange(6):
            plt.plot(show_time, task_paradigms_one_hot[condition, show_time], linewidth=1., linestyle='--', color=colors[condition])
            plt.plot(show_time, task_paradigms_conv[condition, show_time], linewidth=3., color=colors[condition], label=f'Condition {condition}')

        plt.legend(loc='center right')
        plt.title(f'Task paradigm')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

        plt.subplot(412)
        plt.plot(show_time, fmri_data[region, show_time], color='orange', linewidth=3., label='Empirical')
        avg_fmri_data = np.average(fmri_data[region, :], axis=0)
        plt.axhline(avg_fmri_data, color='pink', linewidth=3., linestyle='--', label=f'Empirical average over time ({avg_fmri_data:.3f})')

        plt.legend(loc='upper right')
        plt.title(f'Empirical fMRI timecourse')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.ylim(-3.5, 3.5)

        plt.subplot(413)

    cmap = plt.get_cmap('Blues')
    colors = [cmap(i) for i in np.linspace(0, .7, data_deepko.shape[0])]
    plt.plot(0, 0, color=colors[-1], label='Surrogates')
    for surrogate in np.arange(1, data_deepko.shape[0], 1):
        plt.plot(show_time, data_deepko[surrogate, show_time, region], color=colors[surrogate], linewidth=.7)
    avg_deepko_data = np.average(data_deepko[1:, :, region], axis=0)
    plt.plot(show_time, fmri_data[region, show_time], color='orange', linewidth=3., label='Empirical')
    plt.plot(show_time, avg_deepko_data[show_time], color='black', linewidth=3., label='Surrogates average')
    #plt.axhline(np.average(avg_deepko_data, axis=0), color='pink', linewidth=3., linestyle='--', label=f'Surrogate average over time ({np.average(avg_deepko_data, axis=0):.3f})')
    plt.axhline(0, color='white', linestyle='dotted')

    plt.legend(loc='upper right')
    #plt.title(f'Surrogate timecourses')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.ylim(-3.5, 3.5)

    if False:
        plt.subplot(414)
        time = 100
        show_regions = np.arange(379)

        for surrogate in np.arange(1, data_deepko.shape[0], 1):
            plt.plot(show_regions, data_deepko[surrogate, time, :], linewidth=.7)
        avg_deepko_data = np.average(data_deepko[1:, time, :], axis=0)
        plt.plot(show_regions, avg_deepko_data, color='black', linewidth=3., label='Surrogate average')
        plt.axhline(np.average(avg_deepko_data, axis=0), color='pink', linewidth=3., linestyle='--', label=f'Surrogate average over region ({np.average(avg_deepko_data, axis=0):.3f})')

        plt.legend(loc='upper right')
        plt.title(f'Surrogate fMRI value at time {time}')
        plt.xlabel('Region')
        plt.ylabel('Amplitude')
        plt.ylim(-3.5, 3.5)



    plot_fmri.set_figwidth(20)
    plot_fmri.set_figheight(10)

    #plot_fmri.suptitle(f"Comparison of Empirical and DKO Surrogate timecourses for region {region} (task: {task}, subject: {subject})",fontsize=20)

    plt.savefig(f'timecourses_t{task}_s{subject}_region{region}_epoch{epoch}')

if plot_correlation:
    show_regions = np.arange(0, 10, 1)
    alpha = .05

    plot_corr = plt.figure()
    for show_region in show_regions:
        plt.subplot(2, 5, show_region % 10 + 1)
        plt.bar(0, 0, color='blue', label='Non-significant')
        plt.bar(0, 0, color='orange', label='Significant (Pearson)')

        index_significant = np.where(pearsonr_corr[:, show_region, 1] < alpha)[0]
        barlist = plt.bar(np.arange(pearsonr_corr.shape[0]), pearsonr_corr[:, show_region, 0])
        for idx in index_significant:
            barlist[idx].set_color('orange')
        #plt.bar(np.arange(pearsonr_corr.shape[0])[index_significant], pearsonr_corr[index_significant, show_region, 0], color='orange', label='Significant (Pearson)')

        plt.legend(loc='upper right')
        plt.ylim(0, 1)
        plt.xlabel('Surrogate')
        plt.ylabel('Correlation coefficient')
        plt.title(f'Region {show_region}')

    plot_corr.set_figwidth(20)
    plot_corr.set_figheight(15)
    plot_corr.suptitle(
        f"Correlation between DKO surrogate and empirical fMRI timecourses (task: {task}, subject: {subject}, region: {show_regions[0]}-{show_regions[-1]})",
        fontsize=20)
    plt.savefig(f'correlation_t{task}_s{subject}_region{show_regions[0]}-{show_regions[-1]}')

if plot_heatmap:
    plot_hm = plt.figure()
    sns.heatmap(pearsonr_corr[:, :, 0].T)
    plt.xlabel('Surrogate')
    plt.ylabel('Region')
    plt.title(f'Correlation between DKO surrogate and empirical fMRI timecourses (task: {task}, subject: {subject})')
    plot_hm.set_figwidth(20)
    plot_hm.set_figheight(15)
    plt.savefig(f'correlation_t{task}_s{subject}_heatmap')
print("Done!")


''' 
    knock = scipy.io.loadmat(f'data/output/knockoffs/DeepKO_knock_t{task}_s{subject}.mat')['data']
    knock = np.swapaxes(knock, 1, 2)
    print('knock.shape : ', knock.shape)

    for surrogate in np.arange(0, knock.shape[0], 1):
        plt.plot(np.arange(knock.shape[2]), knock[surrogate, cluster, :], linewidth=.7)
    avg_knock = np.average(knock[1:, cluster, :], axis=0)
    plt.plot(show_time, avg_knock[show_time], color='black', linewidth=3., label='Surrogate average')
    plt.axhline(np.average(avg_knock, axis=0), color='pink', linewidth=3., linestyle='--', label=f'Surrogate average over time ({np.average(avg_knock, axis=0):.3f})')
    plt.legend(loc='upper right')
    plt.title(f'Surrogate timecourse before expand (cluster:{cluster})')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.ylim(-3.5, 3.5)
    
    all_knockoff = scipy.io.loadmat(f'data/output/knockoffs/DeepKO_all_knockoff_t{task}_s{subject}.mat')['data']
    all_knockoff = np.swapaxes(all_knockoff, 1, 2)

    print('all_knockoff.shape : ', all_knockoff.shape)
    for surrogate in np.arange(1, all_knockoff.shape[0], 1):
        plt.plot(np.arange(all_knockoff.shape[2]), all_knockoff[surrogate, region, :], linewidth=.7)
    avg_all_knockoff = np.average(all_knockoff[1:, region, :], axis=0)
    plt.plot(show_time, avg_all_knockoff[show_time], color='black', linewidth=3., label='Surrogate average')
    plt.axhline(np.average(avg_all_knockoff, axis=0), color='pink', linewidth=3., linestyle='--', label=f'Surrogate average over time ({np.average(avg_all_knockoff, axis=0):.3f})')
    plt.legend(loc='upper right')
    plt.title(f'Surrogate timecourse')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.ylim(-3.5, 3.5)
'''