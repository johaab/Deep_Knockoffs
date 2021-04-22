# Deep Knockoffs for fMRI data

## Abstract
One path toward the understanding of brain operation goes by a sufficient comprehension of its structure and activity. Functional magnetic resonance imaging (fMRI) has become the major neuroimaging method used for brain mapping, thanks to its excellent spatial resolution and its non-invasive nature. In this report, the novel application of _Knockoff Filter_ for fMRI data was investigated, which would provide an alternative to the phase randomisation technique widely used on such data. The Knockoff methodology provides the considerable advantage of controlling false discovery rate while performing feature selection. We first concentrated our efforts on analysing the fMRI time course surrogates produced using _Deep Knockoffs_, and then employed those surrogates to construct one-sample nonparametric tests at both the individual and group levels. Our results show that this innovative approach while being promising requires more efforts to achieve meaningful outcomes in the context of brain mapping.

## Data
Data were provided by the [Center for Biomedical Imaging](https://cibm.ch/) (CIBM).

## Context
This project was led as part of the Master in [Computational Biology and Bioinformatics](https://cbb.ethz.ch/) at the [Swiss Federal Institute of Technology in Zurich](https://ethz.ch/en.html) (ETHZ).

## Software dependencies

- python==3.8.5
- numpy==1.19.2
- scipy==1.5.2
- torch==1.7.0
- cvxopt==1.2.5
- cvxpy==1.1.7
- pandas==1.1.3
- matplotlib==3.3.2
- seaborn==0.11.0
- statsmodels==0.12.1
- sklearn
- jupyter
- Cython

## Installation guide

 `pip3 install -r requirements.txt
  pip3 install fanok
  cd deepknockoffs/
  pip3 install DeepKnockoffs/
  pip3 install torch-two-sample-master/`

## File structure
Project
|
|-- data
|   |-- input
|		|-- Glasser360_2mm_codebook.mat
|		|-- hrf.mat
|		|-- TaskParadigms
|		|-- X_tfMRI_EMOTION_LR_Glasser360.mat
|		|-- X_tfMRI_GAMBLING_LR_Glasser360.mat
|		|-- X_tfMRI_LANGUAGE_LR_Glasser360.mat
|		|-- X_tfMRI_MOTOR_LR_Glasser360.mat
|		|-- X_tfMRI_RELATIONAL_LR_Glasser360.mat
|		|-- X_tfMRI_SOCIAL_LR_Glasser360.mat
|		|-- X_tfMRI_WM_LR_Glasser360.mat
|   |-- output
|       |-- beta
|       |-- img
|       |-- knockoffs
|		|-- pearsonr
|
|-- deepknockoffs/
|    |-- torch-two-sample-master/
|
|-- implementation
|    |-- __init__.py
|    |-- glm.py
|    |-- knockoff_class.py
|    |-- load.py
|    |-- non_parametric.py
|    |-- params.py
|    |-- utils.py 
|
|-- PlotGraph/
|
|-- group_analysis.py
|-- plot_correlation.py
|-- plot_test_individual.py
|
|-- .gitignore
|-- __init__.py
|-- README.md


## Authors
Student: [Jonathan Haab](https://www.linkedin.com/in/jonathan-haab/)

Supervisor: [Dr. Maria Giulia Preti](https://miplab.epfl.ch/index.php/people/preti)

Built on the work of Alec Flowers, Alexander Glavackij and Janet van der Graaf (code available [here](https://gitlab.com/aglavac/machine-learning-cs433-p2/-/tree/master)) and the original [Deep Knockoffs implementation](https://github.com/msesia/deepknockoffs)