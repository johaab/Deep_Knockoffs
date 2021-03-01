### This script contains the classes to build and train the knockoffs.

import numpy as np
import pickle
from os.path import join
import abc
import pandas as pd
import matplotlib.pyplot as plt
import fanok
import torch
import scipy.io

from DeepKnockoffs import GaussianKnockoffs, KnockoffMachine
from deepknockoffs.examples.diagnostics import compute_diagnostics, ScatterCovariance

from implementation.params import get_params, ALPHAS
import implementation.load as load
from implementation.utils import KNOCK_DIR, IMG_DIR, BETA_DIR, plot_goodness_of_fit, do_pre_process
from implementation.glm import glm
from implementation.non_parametric import uncorrected_test, corrected_test, get_corrected_betas


class KnockOff(abc.ABC):
    """
    Base class for all knockoffs.
    """

    def __init__(self, task=None, subject=None):
        assert task in ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM'], \
            'Task must be a value in - [EMOTION, GAMBLING, LANGUAGE, MOTOR, RELATIONAL, SOCIAL, WM]'
        assert isinstance(subject, int)
        assert (subject >= 0) & (subject <= 100)

        self.task = task
        self.subject = subject

        self.NAME = None
        self.x_train = None
        self.generator = None
        self.file = None
        self.iters = None

    def load_existing(self, x):
        self.x_train = x

    def load_fmri(self):
        self.x_train = load.load_fmri(task=self.task)
        self.x_train = self.x_train[self.subject]

    def load_paradigms(self):
        paradigms = load.load_task_paradigms(task=self.task)
        paradigms = np.expand_dims(paradigms[self.subject, :], axis=0)
        paradigms = np.repeat(paradigms, self.iters + 1, axis=0)
        return paradigms

    @staticmethod
    def save_pickle(dir, file, to_pickle):
        print(f'Saving file {file}')
        path = join(dir, file)
        with open(path, "wb") as f:
            pickle.dump(to_pickle, f)

    @staticmethod
    def save_mat(dir, file, to_mat):
        # save a file as .mat
        print(f'Saving file {file}')
        path = join(dir, file)
        scipy.io.savemat(path, {'data': to_mat})

    def check_data(self, x=None, transpose=False):
        if x is not None:
            self.x_train = x
        if self.x_train is None:
            ValueError('x cannot be None. Provide data or use load_fmri()')

        if transpose:
            x = self.x_train.T
        else:
            x = self.x_train
        return x

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def generate(self, x):
        pass

    @abc.abstractmethod
    def expand(self, x, groups=None):
        pass

    def transform(self, x=None, iters=100, groups=None, save=False):
        """Build the knockoff"""
        self.iters = iters
        x = self.check_data(x, transpose=True)
        for i in range(iters):
            knock = self.generate(x)
            xk = self.expand(knock, groups)
            if i == 0:
                all_knockoff = np.zeros((iters, xk.shape[0], xk.shape[1]))
            all_knockoff[i, :, :] = xk

        expand_x = self.expand(x, groups)
        all_knockoff = np.concatenate((np.expand_dims(expand_x, axis=0), all_knockoff), axis=0)
        if save:
            self.save_pickle(KNOCK_DIR, self.NAME + '_KO_' + self.file, all_knockoff)
        return all_knockoff

    def diagnostics(self, x=None, n_exams=100):
        """Generate diagnostics to evaluate the performance of the knockoffs of the given method"""
        results = pd.DataFrame(columns=['Method', 'Metric', 'Swap', 'Value', 'Sample'])
        alphas = ALPHAS
        x_train = self.check_data(x, transpose=True)
        # Diagnostics needs an even number of timecourse length
        if x_train.shape[0] % 2 != 0:
            x_train = x_train[:-1, :]
        x_train_tensor = torch.from_numpy(x_train).double()

        for exam in range(n_exams):
            # diagnostics for deep knockoffs
            Xk_train_g = self.generate(x_train)
            Xk_train_g_tensor = torch.from_numpy(Xk_train_g).double()
            new_res = compute_diagnostics(x_train_tensor, Xk_train_g_tensor, alphas, verbose=False)
            new_res["Method"] = self.NAME
            new_res["Sample"] = exam
            results = results.append(new_res)
            if exam == 0:
                ScatterCovariance(x_train, Xk_train_g)
                plt.title(f"Covariance Scatter Plot {self.NAME}")
                file_path = join(IMG_DIR, f"{self.NAME}_t{self.task}_s{self.subject}_scatter_cov.pdf")
                plt.savefig(file_path, format="pdf")
                plt.show()

        print(results.groupby(['Method', 'Metric', 'Swap']).describe())
        for metric, title, swap_equals_self in zip(["Covariance", "KNN", "MMD", "Energy", "Covariance"],
                                                   ["Covariance Goodness-of-Fit", "KNN Goodness-of-Fit",
                                                    "MMD Goodness-of-Fit",
                                                    "Energy Goodness-of-Fit",
                                                    "Absolute Average Pairwise Correlations between Variables and knockoffs"],
                                                   [False, False, False, False, True]):
            plot_goodness_of_fit(results, metric, title, self.NAME, swap_equals_self)

        return results

    def statistic(self, all_knockoff, save=False):
        """Generates beta-values"""
        hrf = load.load_hrf_function()
        paradigms = self.load_paradigms()
        all_knockoff = np.swapaxes(all_knockoff, 1, 2)
        _, _, betas, _, _, _ = glm(all_knockoff, paradigms, hrf)
        if save:
            self.save_mat(BETA_DIR, f"{self.NAME}_KObetas_t{self.task}_s{self.subject}.mat", betas)
        return betas

    def threshold(self, ko_betas, save=False):
        """Thresholds the empirical statistic with Non-Parametric Tests and returns corrected and uncorrected beta-values"""
        beta_path = join(BETA_DIR, f"GLM_betas_{self.task}")
        try:
            true_betas = scipy.io.loadmat(beta_path)['beta'][self.subject, :, :]
        except FileNotFoundError:
            print(f'Need to run GLM to get the true beta value for {self.task}. File name should be - '
                  f'GLM_betas_{self.task}')

        uncorrected = uncorrected_test(ko_betas)
        corrected = corrected_test(ko_betas)

        uncorrected_betas = get_corrected_betas(uncorrected, true_betas)
        corrected_betas = get_corrected_betas(corrected, true_betas)

        if save:
            self.save_mat(BETA_DIR, f"{self.NAME}_uncorrected_betas_t{self.task}_s{self.subject}.mat",
                          uncorrected_betas)
            self.save_mat(BETA_DIR, f"{self.NAME}_corrected_betas_t{self.task}_s{self.subject}.mat", corrected_betas)

        return uncorrected_betas, corrected_betas


class LowRankKnockOff(KnockOff):
    """
    Class to build Low Rank Gaussian Knockoffs (LGKO).

    Attributes
    ----------
    task : string from ['MOTOR', 'GAMBLING', 'RELATIONAL', 'SOCIAL', 'WM', 'EMOTION', 'LANGUAGE']
        Specific task for which the knockoff is built.
    subject : int
        Index of the subject for which the knockoff is built.

    Methods
    -------
    fit(sigma_hat=None, save=False)
        Trains the low rank Gaussian knockoff machine.

    generate()
        Generates knockoffs for the Low Rank Gaussian model.

    expand()
        Expands the knockoffs to original size.

    """

    def __init__(self, task, subject):
        super().__init__(task, subject)
        self.file = f"t{task}_s{subject}.pickle"
        self.NAME = 'LowRankKO'

    def fit(self, x=None, rank=120):
        x = self.check_data(x, transpose=True)
        factor_model = fanok.RandomizedLowRankFactorModel(rank=rank)
        self.generator = fanok.LowRankGaussianKnockoffs(factor_model)
        self.generator.fit(X=x)

    def generate(self, x):
        return self.generator.transform(X=x)

    def expand(self, x, groups=None):
        return x


class GaussianKnockOff(KnockOff):
    """
    Class to build Clustered Gaussian Knockoffs (CGKO).

    Attributes
    ----------
    task : string from ['MOTOR', 'GAMBLING', 'RELATIONAL', 'SOCIAL', 'WM', 'EMOTION', 'LANGUAGE']
        Specific task for which the knockoff is built.
    subject : int
        Index of the subject for which the knockoff is built.

    Methods
    -------
    pre_process(x=None, save=False)
        Preprocesses data by clustering.

    fit(sigma_hat=None, save=False)
        Trains the second-order knockoff machine.

    generate()
        Generates knockoffs for the multivariate Gaussian model.

    expand()
        Expands the knockoffs to original size.

    """

    def __init__(self, task, subject):
        super().__init__(task, subject)
        self.NAME = 'GaussianKO'
        self.max_corr = None
        self.sigma_hat = None
        self.x_train = None
        self.corr_g = None
        self.groups = None

    def pre_process(self, max_corr, x=None, save=False):
        self.max_corr = max_corr
        self.file = f"t{self.task}_s{self.subject}_c{self.max_corr}.pickle"

        x = self.check_data(x)
        self.sigma_hat, self.x_train, self.groups, representatives = do_pre_process(x, self.max_corr)

        if save:
            self.save_pickle(KNOCK_DIR, self.NAME + '_tfMRI_' + self.file, (self.sigma_hat, self.x_train))
            self.save_pickle(KNOCK_DIR, self.NAME + '_mapping_' + self.file, (self.groups, representatives))
        return self.groups

    def fit(self, sigma_hat=None, save=False):
        if sigma_hat is not None:
            self.sigma_hat = sigma_hat
        if self.sigma_hat is None:
            raise ValueError("Sigma Hat cannot be None.")

        # Initialize generator of second-order knockoffs
        self.generator = GaussianKnockoffs(self.sigma_hat, mu=np.zeros((self.sigma_hat.shape[0])), method="sdp")
        # Measure pairwise second-order knockoff correlations
        self.corr_g = (np.diag(self.sigma_hat) - np.diag(self.generator.Ds)) / np.diag(self.sigma_hat)
        print('Average absolute pairwise correlation: %.3f.' % (np.mean(np.abs(self.corr_g))))

        if save:
            self.save_pickle(KNOCK_DIR, self.NAME + '_SecOrd_' + self.file, self.generator)
        return self.corr_g

    def generate(self, x):
        return self.generator.generate(x)

    def expand(self, x, groups=None):
        if groups is not None:
            self.groups = groups
        if self.groups is None:
            raise ValueError('Groups cannot be None')

        xk_full = np.zeros((x.shape[0], self.groups.shape[0]))
        for region, my_group in enumerate(self.groups):
            xk_full[:, region] = x[:, my_group]
        return xk_full


class DeepKnockOff(KnockOff):
    """
    Class to build Deep Knockoffs (DKO).

    Attributes
    ----------
    task : string from ['MOTOR', 'GAMBLING', 'RELATIONAL', 'SOCIAL', 'WM', 'EMOTION', 'LANGUAGE']
        Specific task for which the knockoff is built.
    subject : int
        Index of the subject for which the knockoff is built.
    params :
        Set of parameters to train the neural network.

    Methods
    -------
    pre_process(x=None, save=False)
        Preprocesses data by clustering.

    load_x()
        Loads previously generated data (knockoffs).

    load_params()
        Loads set of parameters.

    load_machine()
        Loads a previously trained machine.

    fit(sigma_hat=None, save=False)
        Trains the deep knockoff machine.

    generate()
        Generates knockoffs for the deep model.

    expand()
        Expands the knockoffs to original size.

    """

    def __init__(self, task, subject, params=None):
        super().__init__(task, subject)
        self.params = params

        self.NAME = 'DeepKO'
        self.file = f"{self.NAME}_t{task}_s{subject}"
        self.groups = None
        self.representatives = None

    def pre_process(self, max_corr, save=False):
        assert self.params is None, 'Params already exists, this would override params.'

        gauss = GaussianKnockOff(self.task, self.subject)
        gauss.load_fmri()
        self.groups = gauss.pre_process(max_corr=max_corr, save=True)
        self.x_train = gauss.x_train
        corr_g = gauss.fit()

        p = self.x_train.T.shape[1]
        n = self.x_train.T.shape[0]
        self.params = get_params(p, n, corr_g)
        if save:
            self.save_pickle(KNOCK_DIR, self.file + '_params', self.params)

    def load_x(self, x):
        self.x_train = x

    def load_params(self, params):
        self.params = params

    def load_machine(self):
        assert self.params is not None, ValueError('Params cannot be None. Please pass in params or run pre-process()')
        checkpoint_name = join(KNOCK_DIR, self.file)
        self.generator = KnockoffMachine(self.params)
        self.generator.load(checkpoint_name)

    def fit(self, x=None, params=None):
        if self.generator is not None:
            raise ValueError('Trained generator already exists')
        if params is not None:
            self.params = params

        x = self.check_data(x, transpose=True)

        checkpoint_name = join(KNOCK_DIR, self.file)
        # Where to print progress information
        logs_name = join(KNOCK_DIR, self.file + "_progress.txt")
        # Initialize the machine
        self.generator = KnockoffMachine(self.params, checkpoint_name=checkpoint_name, logs_name=logs_name)
        # Train the machine
        print("Fitting the knockoff machine...")
        self.generator.train(x)
        return self.generator

    def generate(self, x):
        if self.generator is None:
            raise ValueError("Generator cannot be None. Use load_machine() or fit() to train the generator")
        return self.generator.generate(x)

    def expand(self, x, groups=None):
        if groups is not None:
            self.groups = groups
        if self.groups is None:
            raise ValueError('Groups cannot be None')

        xk_full = np.zeros((x.shape[0], self.groups.shape[0]))
        for region, my_group in enumerate(self.groups):
            xk_full[:, region] = x[:, my_group]
        return xk_full
