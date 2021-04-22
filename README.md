# Deep Knockoffs for fMRI data

## Abstract
One path toward the understanding of brain operation goes by a sufficient comprehension of its structure and activity. Functional magnetic resonance imaging (fMRI) has become the major neuroimaging method used for brain mapping, thanks to its excellent spatial resolution and its non-invasive nature. In this report, the novel application of _Knockoff Filter_ for fMRI data was investigated, which would provide an alternative to the phase randomisation technique widely used on such data. The Knockoff methodology provides the considerable advantage of controlling false discovery rate while performing feature selection. We first concentrated our efforts on analysing the fMRI time course surrogates produced using _Deep Knockoffs_, and then employed those surrogates to construct one-sample nonparametric tests at both the individual and group levels. Our results show that this innovative approach while being promising requires more efforts to achieve meaningful outcomes in the context of brain mapping.

## Data
Data were provided by the [Center for Biomedical Imaging](https://cibm.ch/) (CIBM) and can be downloaded [here](https://www.icloud.com/attachment/?u=https%3A%2F%2Fcvws.icloud-content.com%2FB%2FAWlFx0hqH-wjzzGDOnuOdH3HU4WQAVaLcpnjZG5_ek90DZq6hBQovYiv%2F%24%7Bf%7D%3Fo%3DAi-HT95ZQsI8slxFnirRb8XLFZXE9YTA2UlfPk5vfsF5%26v%3D1%26x%3D3%26a%3DCAogIKHF_Pqr9enwaLDTibNq-s3pqNQKqZi5eHsLsOkmUPMSeBCmqr7U_C4Yprq5qIYvIgEAKgkC6AMA_2XYaXFSBMdThZBaBCi9iK9qJn09qTU_dwLvHvbulpvjlDVB9qvGxAojlsafzVTJgKVLc0cZ0EcaciZ0VJUHMA7V-VPk9WK_G51D89ZRmCfo1La1R8lT_jQIi3fcaW3heg%26e%3D1616603143%26fl%3D%26r%3DA9DFBA32-66AF-43DE-A86C-1E5FFC8C74B3-1%26k%3D%24%7Buk%7D%26ckc%3Dcom.apple.largeattachment%26ckz%3D4E223236-0653-48D1-B0DE-40D79BF7DDFD%26p%3D12%26s%3DDPZXPae9tDNZRXESP02ZVKKXqZk&uk=HfXk3IP7ZPh8seJh2MVXBw&f=DataMLP.zip&sz=734406728).

## Context
This project was led as part of the Master in [Computational Biology and Bioinformatics](https://cbb.ethz.ch/) at the [Swiss Federal Institute of Technology in Zurich](https://ethz.ch/en.html) (ETHZ).

## Authors
Student: [Jonathan Haab](https://www.linkedin.com/in/jonathan-haab/)

Supervisor: [Dr. Maria Giulia Preti](https://miplab.epfl.ch/index.php/people/preti)

Built on the work of Alec Flowers, Alexander Glavackij and Janet van der Graaf (code available [here])(https://gitlab.com/aglavac/machine-learning-cs433-p2/-/tree/master) and the original [Deep Knockoffs implementation](https://github.com/msesia/deepknockoffs)