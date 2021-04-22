import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from implementation.utils import PEARSONR_DIR
from os.path import join

#####################################################################
## Intitialisation: a task and a subject
#####################################################################
# select a task in ['MOTOR', 'GAMBLING', 'RELATIONAL', 'SOCIAL', 'WM', 'EMOTION', 'LANGUAGE']
task = 'MOTOR'
subject = 1
n_epochs = 10

#####################################################################
## Load the correlation matrices
#####################################################################
correlations = []

for epoch in range(n_epochs):
    corr_path = join(PEARSONR_DIR, f"DeepKO_pearsonr_t{task}_s{subject}_epoch{epoch + 1}")
    correlations.append(scipy.io.loadmat(corr_path)['beta'])

correlations = np.array(correlations) # shape (10,100,379,2)

avg_correlations = np.average(correlations[:, :, :, 0], axis=1) # average over surrogates, drop the p-value

#####################################################################
## Plot
#####################################################################

plot_corr = plt.figure()
x = np.linspace(1, n_epochs, n_epochs)
for region in range(avg_correlations.shape[1]):
    #if avg_correlations[n_epochs-1, region] < 0:
        #plt.plot(x, avg_correlations[:, region], label=f'region {region}')
    #else:
        plt.plot(x, abs(avg_correlations[:, region]))


#plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.xticks(x)
plt.ylabel('Correlation coefficent\n(absolute value)')
#plt.title('title here')

plot_corr.set_figwidth(6)
plot_corr.set_figheight(4)

plt.savefig(f'DKO_correlation_epoch1-{n_epochs}')

