from implementation import glm, knockoff_class
from implementation.load import load_pickle
from implementation.utils import KNOCK_DIR

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

#
# Need to be placed at the root of the repository in order to run
#

#####################################################################
## Initialize
#####################################################################
# selecting the same task and subject as in the report
# select condition 0 (aka paradigm)
task = 'MOTOR'
subject = 1
condition = 1

# number of surrogates to generate
N = 100

#####################################################################
## GLM
#####################################################################
run_glm = False

if run_glm:
    # running the GLM on all tasks
    glm.run()

#####################################################################
## Generate
#####################################################################
generate_knockoffs = False

if generate_knockoffs:
    # We can load a previously trained machine
    deepko = knockoff_class.DeepKnockOff(task, subject)
    deepko.pre_process(max_corr=.3, save=True)

    # Loading previously trained machine
    _, x_train = load_pickle(KNOCK_DIR, f'GaussianKO_tfMRI_t{task}_s{subject}_c0.3.pickle')
    groups, _ = load_pickle(KNOCK_DIR, f'GaussianKO_mapping_t{task}_s{subject}_c0.3.pickle')
    params = load_pickle(KNOCK_DIR, f'DeepKO_tMOTOR_s{subject}_params')

    deepko.load_x(x_train)
    deepko.load_params(params)
    deepko.load_machine()

    # generating deep knockoffs (iters gives the number of KO)
    data_deepko = deepko.transform(groups=groups, iters=N)

    # calculating the GLM betas for the knockoffs
    deepko_betas = deepko.statistic(data_deepko, save=True)

    # executing the non-parametric test
    uncorrected_betas_deepko, corrected_betas_deepko = deepko.threshold(deepko_betas, save=True)

#####################################################################
## Load
#####################################################################

### Knockoffs
## Data.shape = (N+1,379,5)
# N+1 = 1 real fMRI + N Knockoffs
# 379 regions
# 5 paradigms => we only select one condition
betas = scipy.io.loadmat('data/output/beta/DeepKO_KObetas_tMOTOR_s1.mat')['data']
emp_betas = betas[0, :, condition]
ko_betas = betas[1:, :, condition]

assert betas.shape[0] == N+1 , "Loaded data and surrogate number specified don't match"

### Original
## Data.shape = (100,379,5)
# 100 subjects => we only select one subject
# 379 voxels
# 5 paradigms => we only select one condition
ori_betas = scipy.io.loadmat('data/output/beta/GLM_betas_MOTOR.mat')['beta'][subject, :, condition]

### Uncorrected, i.e. uncorrected non-parametric test
## Data.shape = (379,5)
# 379 voxels
# 5 paradigms => we only select one condition
uncorrected_betas = scipy.io.loadmat('data/output/beta/DeepKO_uncorrected_betas_tMOTOR_s1.mat')['data'][:,condition]


### Corrected, i.e. corrected non-parametric test
## Data.shape = (379,5)
# 379 voxels
# 5 paradigms => we only select one condition
corrected_betas = scipy.io.loadmat('data/output/beta/DeepKO_corrected_betas_tMOTOR_s1.mat')['data'][:,condition]

### Uncontrolled, i.e. uncorrected parametric test
## Data.shape = (100,379,5)
# 100 subjects => we only select one subject
# 379 voxels
# 5 paradigms => we only select one condition
uncontrolled_betas = scipy.io.loadmat('data/output/beta/GLM_uncontrolled_betas_MOTOR.mat')['beta'][subject,:,condition]

### Controlled, i.e. corrected parametric test
## Data.shape = (100,379,5)
# 100 subjects => we only select one subject
# 379 voxels
# 5 paradigms => we only select one condition
controlled_betas = scipy.io.loadmat('data/output/beta/GLM_controlled_betas_MOTOR.mat')['beta'][subject,:,condition]

#####################################################################
## Plot
#####################################################################
show_voxels = 50
X = np.arange(show_voxels)

betas_selection = plt.figure()

# Parametric testing
plt.subplot(211)
plt.bar(X, ori_betas[:show_voxels], width=.2, color='red', label="Original betas produced by GLM")
plt.bar(X+.2, uncontrolled_betas[:show_voxels], width=.2, color='blue', label="Betas selected after uncorrected parametric testing")
plt.bar(X+.4, controlled_betas[:show_voxels], width=.2, color='green', label="Betas selected after corrected parametric testing (using Bonferroni)")
#plt.bar(X+.6, emp_betas[:show_voxels], width=.2, color='orange', label="Empirical betas representant saved with surrogates")
plt.xlabel('Voxel')
plt.ylabel('Value')
plt.title("Parametric testing")
plt.legend(loc='lower right')

# Non-parametric testing
plt.subplot(212)
plt.bar(X, ori_betas[:show_voxels], width=.2, color='red', label="Original betas produced by GLM")
plt.bar(X+.2, uncorrected_betas[:show_voxels], width=.2, color='blue', label="Betas selected after uncorrected non-parametric test")
plt.bar(X+.4, corrected_betas[:show_voxels], width=.2, color='green', label="Betas selected after corrected non-parametric test (using Min/Max statistics)")
#plt.bar(X+.6, emp_betas[:show_voxels], width=.2, color='orange', label="Empirical betas representant saved with surrogates")
plt.xlabel('Voxel')
plt.ylabel('Value')
plt.title("Non-parametric testing")
plt.legend(loc='lower right')

betas_selection.suptitle(f"Investigation of Betas selection (task:{task}, subject:{subject}, condition:{condition})", fontsize=20)
betas_selection.set_figwidth(15)
betas_selection.set_figheight(15)
betas_selection.savefig('betas_selection')

#############
beta_comparison = plt.figure()
plt.bar(X, ori_betas[:show_voxels], width=.4, color='red', label="Original betas produced by GLM")
plt.bar(X+.4, emp_betas[:show_voxels], width=.4, color='orange', label="Empirical betas representant saved with surrogates")
plt.xlabel('Voxel')
plt.ylabel('Value')
plt.title(f"Original betas vs. Betas chosen as representant \n (task:{task}, subject:{subject}, condition:{condition})")
plt.legend(loc='lower right')
betas_selection.set_figwidth(20)
betas_selection.set_figheight(15)
beta_comparison.savefig('betas_comparison')