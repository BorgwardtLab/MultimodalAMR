# ResMLP training

This folder contains the experiments performed by G.V.
The scripts were run on Python 3.9.16, with the packages contained in the `requirements.txt` file.

The `training_scripts` folder contains the code used to produce the results presented in the paper.
Specifically:

- `train_ResAMR_classifier.py` contains the code to train a ResMLP model on the DRIAMS datasets for the predictions tasks.
- `train_recommender_ResAMR.py` is the script used to train the model for recommendation purposes.
- `train_SplitPCA_LR_classifier.py` and `train_JointPCA_LR_classifier.py` contain the training of the PCA+LR baseline. They respectively apply the PCA step before and after concatenation.
- `train_1hotSpecies_ResAMR_classifier.py` is the ablation experiment where the MALDI-TOF spectrum is substituted by the 1-hot encoding of the pathogen species.
- `train_ResAMR_classifier_single_species_comparison.py` trains the models for the comparison with the single drug-single species from the Weis et al. paper.

