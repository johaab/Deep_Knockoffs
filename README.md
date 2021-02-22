# Deep Knockoffs for fMRI data

## Project Description
fMRI is a powerful tool to study the brain and relies on the difference in magnetic properties of arterial and venous blood. Blood-oxygen-level dependent (BOLD) contrast is typically used as a measure of the brain activation level, in other words oxygen-rich blood location indicates active brain regions. By detecting correlations between brain activity and the task performed, one can identify regions linked to critical functions. The relatively weak BOLD response and the presence of noise cause brain mapping to be a challenging task.

In this project, a new framework called _Knockoff Filter_ will be applied to fMRI data in order to find the brain regions exhibiting significant activation during specific tasks. The knockoff methodology strength lays in its control of false discovery rate (FDR) while operating feature selection. The idea behind this method is to generate knockoff features and use those as negative controls. The true importance of a feature can then be deduced by comparing its predic- tive power to the one of its knockoff copy. In the case of fMRI data, the selected features are associated with activated regions of the brain.

The aim of this project is twofold. Firstly, the goal is to deepen the under- standing of the Deep Knockoffs (DKO) setting which relies on a neural network (NN) to produce the knockoffs. The problem faced is as follow: knockoff copies of the features must follow the same distribution as the original features, but this precise distribution is unknown. Thus, the NN has to learn the distribution of the features by iteratively generating knockoffs and updating the network’s parameters based on the compliance with the exchangeability property. At the end of the training, the NN should be able to generate approximate knock- offs copies for the new feature observations drawn from the same underlying distribution. Secondly, the plan is to extend the analysis from individual- to group-based data. Group analysis is conventionally performed in most clinical studies aiming at characterizing functional activations in different groups and assessing differences between them.

## Data
Data were provided by the [Center for Biomedical Imaging](https://cibm.ch/) (CIBM) and can be downloaded [here](https://www.icloud.com/attachment/?u=https%3A%2F%2Fcvws.icloud-content.com%2FB%2FAWlFx0hqH-wjzzGDOnuOdH3HU4WQAVaLcpnjZG5_ek90DZq6hBQovYiv%2F%24%7Bf%7D%3Fo%3DAi-HT95ZQsI8slxFnirRb8XLFZXE9YTA2UlfPk5vfsF5%26v%3D1%26x%3D3%26a%3DCAogIKHF_Pqr9enwaLDTibNq-s3pqNQKqZi5eHsLsOkmUPMSeBCmqr7U_C4Yprq5qIYvIgEAKgkC6AMA_2XYaXFSBMdThZBaBCi9iK9qJn09qTU_dwLvHvbulpvjlDVB9qvGxAojlsafzVTJgKVLc0cZ0EcaciZ0VJUHMA7V-VPk9WK_G51D89ZRmCfo1La1R8lT_jQIi3fcaW3heg%26e%3D1616603143%26fl%3D%26r%3DA9DFBA32-66AF-43DE-A86C-1E5FFC8C74B3-1%26k%3D%24%7Buk%7D%26ckc%3Dcom.apple.largeattachment%26ckz%3D4E223236-0653-48D1-B0DE-40D79BF7DDFD%26p%3D12%26s%3DDPZXPae9tDNZRXESP02ZVKKXqZk&uk=HfXk3IP7ZPh8seJh2MVXBw&f=DataMLP.zip&sz=734406728).

## Context
This project was led as part of the Master in [Computational Biology and Bioinformatics](https://cbb.ethz.ch/) at the [Swiss Federal Institute of Technology in Zurich](https://ethz.ch/en.html) (ETHZ).

## Authors
Student: [Jonathan Haab](https://www.linkedin.com/in/jonathan-haab/)

Supervisor: [Dr. Maria Giulia Preti](https://miplab.epfl.ch/index.php/people/preti)

Built on the work of [Alec Flowers, Alexander Glavackij and Janet van der Graaf](https://gitlab.com/aglavac/machine-learning-cs433-p2/-/tree/master)