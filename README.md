
# Robust Uncertainty Estimation in Medical Machine Learning Applications with the Laplace Approximation
Code for my Master's Thesis "Robust Uncertainty Estimation in Medical Machine Learning Applications with the Laplace Approximation". 


## How this repository is structured
- My experimentation used [laplace-redux](https://github.com/runame/laplace-redux) as a starting point. This was heavily modified to implement ``Weight Included Temperature Scaling`` and ``Covariance Scaling``, and also to employ new datasets and train new models.
- The SkinLesions dataset was prepared following the instructions from [Robust-Skin-Lesion-Classification](https://github.com/ZerojumpLine/Robust-Skin-Lesion-Classification) (Instructions in [Robust-Skin-Lesion-Classification/skinlesiondatasets](Robust-Skin-Lesion-Classification/skinlesiondatasets/README.md)).
- Experimentation was done on a SLURM cluster. The scripts and commands used are in [SlurmClusterScripts](SlurmClusterScripts). With minimal adaption they should be possible to use in any environment. 
- All scripts used for plotting are in [PlottingScripts](PlottingScripts). They have not been improved for clarity, so they are relatively convoluted. They require all the data from running the experiments. Most of them have to be started from a jupyter-notebook running in the ``laplace-redux`` directory for them to work.  


## Replicating my results
When wanting to replicate my work, one should start by obtaining the SkinLesions dataset. Then the training scripts and experimentation scripts should be run and afterwards the plotting should be done. 
