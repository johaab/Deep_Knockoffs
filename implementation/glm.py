import implementation.load as load
import numpy as np
import statsmodels.api as sm
from implementation.utils import BETA_DIR


def glm(fMRI, task_paradigms, hrf):
    """
    Computes the General Linear Model (GLM) from fMRI data to estimate the parameters for a given task

        Parameters:
        ----------
        fMRI: fMRI BOLD signal which is a 3-d array of size (n_subjects, n_regions, n_timepoints)
        task_paradigms: temporal details on the presentation of the tasks for each subject, with size (n_subjects, n_timepoints)
        hrf: Hemodynamic Response Function, used to convolute the task paradigm

        Return:
        ----------
        act: 2-d array of size (n_subjects, n_regions) with {0, 1} values corresponding to activation of
                    brain regions according to the result of the GLM
        betas: 2-d array of size (n_subjects, n_regions) with beta values resulting from GLM
    """
    assert fMRI.shape[1] == 379, 'Expect to see 379 brain regions'
    # Reshaping so that fMRI and task_paradigms shapes match by keeping the shorter length
    if fMRI.shape[2] != task_paradigms.shape[1]:
        fMRI = fMRI[:, :, :min(fMRI.shape[2], task_paradigms.shape[1])]
        task_paradigms = task_paradigms[:, :min(fMRI.shape[2], task_paradigms.shape[1])]

    assert fMRI.shape[2] == task_paradigms.shape[1], \
        f"fMRI and task_paradigms shapes do not match: {fMRI.shape[1]} and {task_paradigms.shape}"

    # do one hot encoding
    task_paradigms_one_hot = load.separate_conditions(task_paradigms)

    # do the convolution
    task_paradigms_conv = load.do_convolution(task_paradigms_one_hot, hrf)

    # fit the glm for every subject and region
    print(f"Fitting GLM for {fMRI.shape[0]} subjects and {fMRI.shape[1]} regions...")
    p_value = 0.05
    bonferonni_value = p_value / fMRI.shape[1]
    activations = np.zeros((task_paradigms_one_hot.shape[0], fMRI.shape[1], task_paradigms_one_hot.shape[1] - 1))
    controlled_act = np.zeros((task_paradigms_one_hot.shape[0], fMRI.shape[1], task_paradigms_one_hot.shape[1] - 1))
    betas = np.zeros((task_paradigms_one_hot.shape[0], fMRI.shape[1], task_paradigms_one_hot.shape[1] - 1))
    tvalues = np.zeros((task_paradigms_one_hot.shape[0], fMRI.shape[1], task_paradigms_one_hot.shape[1] - 1))

    for subject in range(fMRI.shape[0]):
        for region in range(fMRI.shape[1]):
            X = np.swapaxes(task_paradigms_conv[subject, 1:, :], 0, 1)
            y = fMRI[subject, region, :]
            mod = sm.OLS(y, X)
            res = mod.fit()
            p_values = res.pvalues
            coef = res.params
            tval = res.tvalues
            # prints RuntimeError when y==0, i.e. all coefficients of the OLS are zero
            activations[subject, region, :] = p_values < p_value
            controlled_act[subject, region, :] = p_values < bonferonni_value
            betas[subject, region, :] = coef
            tvalues[subject, region, :] = tval
    uncontrolled_betas = activations * betas
    controlled_betas = controlled_act * betas

    print("Done!")

    return activations, controlled_act, betas, tvalues, uncontrolled_betas, controlled_betas



def run():
    """
    Runs the GLM for the tasks defined
    """
    tasks = ['MOTOR', 'GAMBLING', 'RELATIONAL', 'SOCIAL', 'WM', 'EMOTION', 'LANGUAGE']
    hrf = load.load_hrf_function()

    for task in tasks:
        # loading data for a specific task
        print(f'============================== \n {task} \n==============================')
        print(f"Loading data for task {task}...")
        fMRI = load.load_fmri(task)
        task_paradigms = load.load_task_paradigms(task)

        # computing glm for a specific task
        print(f'Computing GLM for task {task}...')
        activations, image_act, betas, tvalues, uncontrolled_betas, controlled_betas = glm(fMRI, task_paradigms, hrf)

        # saving output for a specific task
        print(f"Saving activations and beta values for task {task}...")
        # load.save_pickle(activations, BETA_DIR, 'activation', task)
        # load.save_pickle(betas, BETA_DIR, 'betas', task)
        load.save_mat(betas, BETA_DIR, 'GLM_betas', task)
        load.save_mat(uncontrolled_betas, BETA_DIR, 'GLM_uncontrolled_betas', task)
        # load.save_pickle(uncontrolled_betas, BETA_DIR, 'GLM_uncontrolled_betas', task)

        load.save_mat(controlled_betas, BETA_DIR, 'GLM_controlled_betas', task)
        # load.save_pickle(controlled_betas, BETA_DIR, 'GLM_controlled_betas', task)

