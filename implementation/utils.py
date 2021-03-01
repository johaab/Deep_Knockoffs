### This script contains utilities.

import pathlib
import matplotlib.pyplot as plt
import matplotlib
import os.path
import numpy as np
import scipy.cluster.hierarchy as spc

REPO_ROOT = pathlib.Path(__file__).absolute().parents[1].absolute().resolve()
assert (REPO_ROOT.exists())
DATA_DIR = (REPO_ROOT / "data").absolute().resolve()
assert (DATA_DIR.exists())
INPUT_DIR = (DATA_DIR / "input").absolute().resolve()
assert (INPUT_DIR.exists())
OUTPUT_DIR = (DATA_DIR / "output").absolute().resolve()
assert (OUTPUT_DIR.exists())
KNOCK_DIR = (OUTPUT_DIR / "knockoffs").absolute().resolve()
assert (KNOCK_DIR.exists())
IMG_DIR = (OUTPUT_DIR / "img").absolute().resolve()
assert (IMG_DIR.exists())
BETA_DIR = (OUTPUT_DIR / "beta").absolute().resolve()
assert (BETA_DIR.exists())


def plot_goodness_of_fit(results, metric, title, name, swap_equals_self=False, save_img=True):
    """Plots Goodness Of Fit"""
    if not swap_equals_self:
        file = f"{name}_box_{metric}.pdf"
    else:
        file = f"{name}_box_corr.pdf"
    fig, ax = plt.subplots(figsize=(12, 6))
    do_plot(results, metric, swap_equals_self, ax)
    ax.set_title(title)
    file_path = os.path.join(IMG_DIR, file)
    if save_img:
        plt.savefig(file_path, format="pdf")
    return fig, ax


def do_plot(results, metric, swap_equals_self, ax):
    import seaborn as sns
    if not swap_equals_self:
        data = results[(results.Metric == metric) & (results.Swap != "self")]
    else:
        data = results[(results.Metric == metric) & (results.Swap == "self")]
    sns.boxplot(x="Swap", y="Value", hue="Method", data=data, ax=ax)
    return ax


def compare_diagnostics(results):
    """
    Plots the diagnostics for all `Methods` in the results data frame. Allows for direct
    comparison between knockoff generators.
    """
    # init
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (16, 10),
              'axes.labelsize': '20',
              'axes.titlesize': '20',
              'xtick.labelsize': '20',
              'ytick.labelsize': '20'}
    matplotlib.rcParams.update(params)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
    axs = axs.flatten()
    # plotting offdiagonal Covariance diagnostics
    do_plot(results, 'Covariance', False, axs[0])
    axs[0].axhline(0, linestyle='--', c='red')
    axs[0].set_title('Offdiagonal Covariance\nGoodness-of-Fit')
    axs[0].get_legend().remove()
    # plotting KNN diagnostics
    do_plot(results, 'KNN', False, axs[1])
    axs[1].axhline(0.5, linestyle='--', c='red')
    axs[1].set_title('KNN Goodness-of-Fit')
    axs[1].get_legend().remove()
    # plotting MMD diagnostics
    do_plot(results, 'MMD', False, axs[2])
    axs[2].axhline(0, linestyle='--', c='red')
    axs[2].set_title('MMD Goodness-of-Fit')
    axs[2].get_legend().remove()
    # plotting Energy diagnostics
    do_plot(results, 'Energy', False, axs[3])
    axs[3].axhline(0, linestyle='--', c='red')
    axs[3].set_title('Energy Goodness-of-Fit')
    axs[3].get_legend().remove()
    # plotting diagonal Covariance diagnotics
    do_plot(results, 'Covariance', True, axs[4])
    axs[4].axhline(0, linestyle='--', c='red')
    axs[4].set_title('Diagonal Covariance\nGoodness-of-Fit')
    axs[4].get_legend().remove()
    # axs[4].tick_params(**params)
    fig.delaxes(ax=axs[5])
    fig.tight_layout()
    handles, labels = axs[4].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize='xx-large', bbox_to_anchor=(0.95, 0.2))
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def do_pre_process(X, max_corr):
    """
    Performs pre-processing by clustering.
    :param X: data
    :param max_corr: maximum correlation for which data will be clustered
    :return:
        SigmaHat_repr: Sigma Hat matrix for group representatives
        X_repr: data representatives
        groups: clusters of the representatives
        representatives: contains one representative per each cluster
    """
    from deepknockoffs.examples import data
    # calc SigmaHat
    SigmaHat = np.cov(X)
    Corr = data.cov2cor(SigmaHat)
    # Compute distance between variables based on their pairwise absolute correlations
    pdist = spc.distance.squareform(1 - np.abs(Corr))
    # Apply average-linkage hierarchical clustering
    linkage = spc.linkage(pdist, method='average')
    corr_max = max_corr
    d_max = 1 - corr_max
    # Cut the dendrogram and define the groups of variables
    groups = spc.cut_tree(linkage, height=d_max).flatten()
    print("Divided " + str(len(groups)) + " variables into " + str(np.max(groups) + 1) + " groups.")
    linkage = spc.linkage(pdist, method='average')
    print("Divided " + str(len(groups)) + " variables into " + str(np.max(groups) + 1) + " groups.")
    # Plot group sizes
    _, counts = np.unique(groups, return_counts=True)
    print("Size of largest groups: " + str(np.max(counts)))
    print("Mean groups size: " + str(np.mean(counts)))
    # Pick one representative for each cluster
    representatives = np.array([np.where(groups == g)[0][0] for g in
                                np.arange(np.max(groups) + 1)])  # + 1 due to np.arange(), bug in original code
    # Sigma Hat matrix for group representatives
    SigmaHat_repr = SigmaHat[representatives, :][:, representatives]
    # Correlations for group representatives
    Corr_repr = data.cov2cor(SigmaHat_repr)
    # fMRI representatives
    X_repr = X[representatives]
    print(f"Eigenvalue for Sigma Hat, Min: {np.min(np.linalg.eigh(SigmaHat)[0])}")
    print(f"Eigenvalue for Sigma Hat Representatives, Min: {np.min(np.linalg.eigh(SigmaHat_repr)[0])}")
    print(f"Original for Correlations, Max: {np.max(np.abs(Corr - np.eye(Corr.shape[0])))}")
    print(f"Representatives for Correlations, Max: {np.max(np.abs(Corr_repr - np.eye(Corr_repr.shape[0])))}")
    return SigmaHat_repr, X_repr, groups, representatives
