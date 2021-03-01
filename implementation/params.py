# CONSTANTS

ALPHAS = [1., 2., 4., 8., 16., 32., 64., 128.]


def get_params(p, n, n_corr):
    # Set the parameters for training deep knockoffs
    pars = dict()
    # Number of epochs
    pars['epochs'] = 5
    # Number of iterations over the full data per epoch
    pars['epoch_length'] = 100
    # data type, either "continuous" or "binary"
    pars['family'] = "continuous"
    # Dimensions of the data
    pars['p'] = p
    # Size of the test set
    pars['test_size'] = 0
    # Batch size
    pars['batch_size'] = int(0.5 * n)
    # Learning rate
    pars['lr'] = 0.01
    # When to decrease learning rate (unused when equal to number of epochs)
    pars['lr_milestones'] = [pars['epochs']]
    # Width of the network (number of layers is fixed to 6)
    pars['dim_h'] = int(10 * p)
    # Penalty for the MMD distance
    pars['GAMMA'] = 1.0
    # Penalty encouraging second-order knockoffs
    pars['LAMBDA'] = 0.1
    # Decorrelation penalty hyperparameter
    pars['DELTA'] = 0.1
    # Target pairwise correlations between variables and knockoffs
    pars['target_corr'] = n_corr
    # Kernel widths for the MMD measure (uniform weights)
    pars['alphas'] = [1., 2., 4., 8., 16., 32., 64., 128.]
    return pars
