import implementation.load as load
import numpy as np
import statsmodels.api as sm
import scipy.io
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



def run(selected_task='all'):
    """
    Runs the GLM for the tasks defined
    """
    tasks = ['MOTOR', 'GAMBLING', 'RELATIONAL', 'SOCIAL', 'WM', 'EMOTION', 'LANGUAGE']
    hrf = load.load_hrf_function()

    if selected_task != 'all':
        assert selected_task in tasks, "Task must be a value in - [EMOTION, GAMBLING, LANGUAGE, MOTOR, RELATIONAL, SOCIAL, WM] or 'all' "
        tasks = [selected_task]

    for task in tasks:
        # loading data for a specific task
        print(f'============================== \n {task} \n==============================')
        print(f"Loading data for task {task}...")
        fmri = load.load_fmri(task)
        task_paradigms = load.load_task_paradigms(task)

        # computing glm for a specific task
        print(f'Computing GLM for task {task}...')
        activations, image_act, betas, tvalues, uncorrected_betas, corrected_betas = glm(fmri, task_paradigms, hrf)

        # extract rest fMRI
        #rest_fMRI = extract_rest(fmri, task_paradigms)

        # saving output for a specific task
        print(f"Saving activations and beta values for task {task}...")
        # load.save_pickle(activations, BETA_DIR, 'activation', task)
        # load.save_pickle(betas, BETA_DIR, 'betas', task)
        load.save_mat(betas, BETA_DIR, 'GLM_betas', task)
        load.save_mat(uncorrected_betas, BETA_DIR, 'GLM_uncorrected_betas', task)
        # load.save_pickle(uncontrolled_betas, BETA_DIR, 'GLM_uncontrolled_betas', task)

        load.save_mat(corrected_betas, BETA_DIR, 'GLM_corrected_betas', task)
        # load.save_pickle(controlled_betas, BETA_DIR, 'GLM_controlled_betas', task)

        #load.save_mat(rest_fMRI, BETA_DIR, 'rest_state_fMRI', task)


def extract_rest(fMRI, task_paradigms):
    assert fMRI.shape[1] == 379, 'Expect to see 379 brain regions'
    # Reshaping so that fMRI and task_paradigms shapes match by keeping the shorter length
    if fMRI.shape[2] != task_paradigms.shape[1]:
        fMRI = fMRI[:, :, :min(fMRI.shape[2], task_paradigms.shape[1])]
        task_paradigms = task_paradigms[:, :min(fMRI.shape[2], task_paradigms.shape[1])]

    assert fMRI.shape[2] == task_paradigms.shape[1], \
        f"fMRI and task_paradigms shapes do not match: {fMRI.shape[1]} and {task_paradigms.shape}"

    # do one hot encoding
    task_paradigms_one_hot = load.separate_conditions(task_paradigms)

    # not sure we need the convolution here
    # do the convolution
    #task_paradigms_conv = load.do_convolution(task_paradigms_one_hot, hrf)

    fMRI_rest = np.empty(fMRI.shape)

    for subject in range(fMRI.shape[0]):
        for region in range(fMRI.shape[1]):
            fMRI_rest[subject, region, :] = task_paradigms_one_hot[subject, 0, :] * fMRI[subject, region, :]

    return fMRI_rest

def run_first_level(selected_task='all', fmri=None, id=None):
    '''
    Run GLM on fMRI timecourses
    '''
    tasks = ['MOTOR', 'GAMBLING', 'RELATIONAL', 'SOCIAL', 'WM', 'EMOTION', 'LANGUAGE']

    if selected_task != 'all':
        assert selected_task in tasks, "Task must be a value in - [EMOTION, GAMBLING, LANGUAGE, MOTOR, RELATIONAL, SOCIAL, WM] or 'all' "
        tasks = [selected_task]

    if fmri is None:
        raise ValueError("fMRI time course need to be provided to run first-level GLM")
    if id is None:
        print("Data ID wasn't specified, it is advised to specify if GLM is run on empirical or surrogate fMRI timecourse")

    hrf = load.load_hrf_function()

    ''' Produces first-level betas '''
    print(f'============================== \n First-level GLM \n==============================')
    for task in tasks:
        # loading data for a specific task
        #print(f"Loading data for task {task}...")
        #fmri = load.load_fmri(task)
        task_paradigms = load.load_task_paradigms(task)

        # computing glm for a specific task
        print(f'Computing GLM for task {task}...')
        activations, image_act, betas, tvalues, uncorrected_betas, corrected_betas = glm(fmri, task_paradigms, hrf)

        # saving output for a specific task
        print(f"Saving activations and beta values for task {task}...")
        # load.save_pickle(activations, BETA_DIR, 'activation', task)
        # load.save_pickle(betas, BETA_DIR, 'betas', task)
        load.save_mat(betas, BETA_DIR, id + '_betas', task)
        load.save_mat(uncorrected_betas, BETA_DIR, id + '_uncorrected_betas', task)
        # load.save_pickle(uncontrolled_betas, BETA_DIR, 'GLM_uncontrolled_betas', task)

        load.save_mat(corrected_betas, BETA_DIR, id + '_corrected_betas', task)
        # load.save_pickle(controlled_betas, BETA_DIR, 'GLM_controlled_betas', task)


def run_second_level(selected_subjects=None, selected_task='all'):
    print(f"Performing second-level GLM...")

    if selected_subjects is None:
        raise ValueError('The group of subjects must be specified')

    tasks = ['MOTOR', 'GAMBLING', 'RELATIONAL', 'SOCIAL', 'WM', 'EMOTION', 'LANGUAGE']

    if selected_task != 'all':
        assert selected_task in tasks, "Task must be a value in - [EMOTION, GAMBLING, LANGUAGE, MOTOR, RELATIONAL, SOCIAL, WM] or 'all' "
        tasks = [selected_task]


    for task in tasks:
        ## Second-level GLM on Empirical data
        empirical_fl_betas = scipy.io.loadmat(f'data/output/beta/GLM_betas_t{task}.mat')['beta'][selected_subjects]

        empirical_sl_betas, uncorrected_empirical_sl_betas, corrected_empirical_sl_betas = sl_glm(empirical_fl_betas)

        print(f"Saving empirical second-level beta values for analysis on task {task}...")

        # saving output for a specific task
        # load.save_pickle(activations, BETA_DIR, 'group_activation', task)
        # load.save_pickle(empirical_sl_betas, BETA_DIR, 'group_GLM_betas', task)
        load.save_mat(empirical_sl_betas, BETA_DIR, 'empirical_sl_betas', task)
        load.save_mat(uncorrected_empirical_sl_betas, BETA_DIR, 'empirical_uncorrected_sl_betas', task)
        # load.save_pickle(uncorrected_empirical_sl_betas, BETA_DIR, 'group_GLM_uncorrected_betas', task)
        load.save_mat(corrected_empirical_sl_betas, BETA_DIR, 'empirical_corrected_sl_betas', task)
        # load.save_pickle(corrected_empirical_sl_betas, BETA_DIR, 'group_GLM_corrected_betas', task)
        print("Done!")
        ## Second-level GLM on Surrogate data
        surrogate_fl_betas = []
        for subject in selected_subjects:
            fl_betas = scipy.io.loadmat(f'data/output/beta/DeepKO_betas_t{task}_s{subject}.mat')['data']
            fl_betas = fl_betas[1:, :, :]  # drop the empirical data
            surrogate_fl_betas.append(fl_betas)

        surrogate_fl_betas = np.array(surrogate_fl_betas)

        surrogate_sl_betas = np.empty([surrogate_fl_betas.shape[1], surrogate_fl_betas.shape[2], surrogate_fl_betas.shape[3]])
        for surrogate in range(surrogate_fl_betas.shape[1]):
            sl_betas, _, _ = sl_glm(surrogate_fl_betas[:, surrogate, :, :])

            surrogate_sl_betas[surrogate, :, :] = sl_betas


        print(f"Saving surrogate second-level beta values for analysis on task {task}...")
        load.save_mat(surrogate_sl_betas, BETA_DIR, 'DeepKO_sl_betas', task)
        print("Done!")


def sl_glm(fl_betas=None):
    assert fl_betas is not None, 'First-level betas must be given to run second-level GLM'

    # Switch shape from (Subjects/Surrogates, Regions, Condition) to (Regions, Condition, Subjects/Surrogates)
    fl_betas = np.moveaxis(fl_betas, 0, -1)

    p_value = 0.05
    bonferonni_value = p_value / fl_betas.shape[0]

    activations = np.zeros([fl_betas.shape[0], fl_betas.shape[1]])
    corrected_act = np.zeros([fl_betas.shape[0], fl_betas.shape[1]])
    sl_betas = np.zeros([fl_betas.shape[0], fl_betas.shape[1]])
    tvalues = np.zeros([fl_betas.shape[0], fl_betas.shape[1]])

    for region in range(fl_betas.shape[0]):
        for condition in range(fl_betas.shape[1]):
            # we want to fit the baseline only
            X = np.ones(fl_betas.shape[2])
            # take the betas for a specific region and task condition across all subjects
            y = fl_betas[region, condition, :]
            mod = sm.OLS(y, X)
            res = mod.fit()

            p_values = res.pvalues
            coef = res.params
            tval = res.tvalues
            # prints RuntimeError when y==0, i.e. all coefficients of the OLS are zero
            activations[region, condition] = p_values < p_value
            corrected_act[region, condition] = p_values < bonferonni_value
            sl_betas[region, condition] = coef
            tvalues[region, condition] = tval
    uncorrected_sl_betas = activations * sl_betas
    corrected_sl_betas = corrected_act * sl_betas

    return sl_betas, uncorrected_sl_betas, corrected_sl_betas

