### This script contains functions necessary for the loading and saving

import os
from collections import OrderedDict
import scipy.io
import numpy as np
import pickle
from implementation.utils import INPUT_DIR


def load_fmri(task='MOTOR'):
    """Load the fMRI BOLD signal which is a 3-d array."""
    assert task in ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM'], \
        'Task must be a value in - [EMOTION, GAMBLING, LANGUAGE, MOTOR, RELATIONAL, SOCIAL, WM]'
    filename = os.path.join(INPUT_DIR, 'X_tfMRI_' + task + '_LR_Glasser360.mat')

    data = scipy.io.loadmat(filename)
    data = data['X']
    # getting data into the following format: [subject, region, timeseries]
    data = np.swapaxes(data, 1, 2)
    data = np.swapaxes(data, 0, 1)

    print(f'Loaded Data - Shape: {data.shape}')
    return data


def load_task_paradigms(task='MOTOR'):
    """Load all the task paradigms for each subject."""
    assert task in ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM'], \
        'Task must be a value in - [EMOTION, GAMBLING, LANGUAGE, MOTOR, RELATIONAL, SOCIAL, WM]'
    DIRECTORY = os.path.join(INPUT_DIR, 'TaskParadigms')
    FILE = task + '_LR.mat'

    regressor = {}
    for filename in os.listdir(DIRECTORY):
        if filename.endswith(FILE):
            task = scipy.io.loadmat(os.path.join(DIRECTORY, filename))
            regressor[filename.split('_')[0]] = task['Regressor']

    regressor = OrderedDict(sorted(regressor.items()))
    # Setting all the subjects to the same number of timepoints
    min_length = (min(regressor.items(), key=lambda x: x[1].shape[1])[1]).shape[1]
    for key, value in regressor.items():
        regressor[key] = value[:, :min_length]
    regressor = np.array(list(regressor.values())).squeeze()

    print(f'Loaded Task Paradigms - Shape: {regressor.shape}')
    return regressor


def load_hrf_function():
    """Load the hrf function"""
    filename = os.path.join(INPUT_DIR, 'hrf.mat')
    hrf = scipy.io.loadmat(filename)['hrf'].squeeze()
    return hrf


def separate_conditions(task_paradigms):
    """
    Separates the conditions given in task_paradigms
    :param task_paradigms: nd_array(n_subjects, n_timeseries)
    :return: nd_array(n_subjects, n_conditions, n_timeseries)
    """
    print("Separating conditions...")
    n_conditions = np.max(task_paradigms) + 1
    task_paradigms_one_hot = np.zeros((task_paradigms.shape[0], n_conditions, task_paradigms.shape[1]))
    for subject in range(task_paradigms.shape[0]):
        task_paradigms_one_hot[subject] = do_one_hot(task_paradigms[subject].squeeze())
    print("Done!")
    return task_paradigms_one_hot


def do_one_hot(l):
    """Performs one-hot encoding"""
    b = np.zeros((l.size, l.max() + 1))
    b[np.arange(l.size), l] = 1
    return b.transpose()


def do_convolution(task_paradigms_one_hot, hrf):
    """
    Performs the convolution with a HRF and the seperate task paradigms.
    :param task_paradigms_one_hot: nd_array(n_subjects, n_conditions, n_timeseries)
    :param hrf: the HRF to perform the conv. with
    :return: nd_array(n_subjects, n_conditions, n_timeseries)
    """
    print("Convolving...")
    task_paradigms_conv = np.zeros(task_paradigms_one_hot.shape)
    for subject in range(task_paradigms_one_hot.shape[0]):
        for condition in range(task_paradigms_one_hot.shape[1]):
            convolution = np.convolve(task_paradigms_one_hot[subject, condition, :], hrf, "full")
            # convolution contains len(hrf) + len(task_paradigms_one_hot[subject, condition, :]) + 1 elements, needs to
            # reduce size to len(ask_paradigms_one_hot[subject, condition, :]) as given in Giulias code fragment
            task_paradigms_conv[subject, condition, :] = convolution[:task_paradigms_one_hot.shape[2]]
    print("Done!")
    return task_paradigms_conv


def save_pickle(data, path, preface, task):
    """Save a file as .pickle"""
    filename = os.path.join(path, f'{preface}_{task}.pickle')
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def save_mat(data, path, preface, task):
    """Save a file as .mat"""
    filename = os.path.join(path, f'{preface}_{task}.mat')
    scipy.io.savemat(filename, {'beta': data})


def load_pickle(path, file):
    """Load pickle file"""
    file_path = os.path.join(path,file)
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data
