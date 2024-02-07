# A multimodal approach to antimicrobial resistance prediction for drug recommendation and large-scale classification.


This repository contains the code used to conduct experiments for the paper [_Multimodal learning in clinical proteomics: enhancing antimicrobial resistance prediction models with chemical information_](https://doi.org/10.1093/bioinformatics/btad717) from Giovanni Visonà, Diane Duroux, Lucas Miranda, Emese Sükei, Yiran Li, Karsten Borgwardt, and Carlos Oliver.

The paper present work aimed at predictiong Antimicrobial Resistance (AMR) using proteomic and chemical features, which is examined through two clinically relevant tasks: drug recommendation and generalized resistance prediction. We demonstrate how the multi-view representation of the pathogenic samples adopted paired with the use of deep learning to  leverage the scale of the available datasets substantially outperfors the previous single-drug and single-species predictive models.

![ROC and PR curves](images/curves.png "Comparison of ROC and PR curves for a few pathogen-antimicrobial combinations, that show the potential of deep learning multimodal models.")


The experiments are organized based on the authors that performed them, while the general utilities and processed data are contained in the `data_split` and `processed_data` forlders. 
`lm_experiments` contains scripts to evaluate the performance of some baseline machine learning models on individual drug-species subsets of the DRIAMS dataset.
`dd_experiments` includes the training of the Siamese Network model and the evaluation of all the recommendation models except the ResMLP.
`gv_experiments` contains scripts to train the baseline PCA+LR experiments and the scripts related to the ResMLP experiments, including classification, recommendation, and ablation experiments.

The MALDI-TOF spectra for reproducing the experiments can be obtained by downloading the DRIAMS dataset files from [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q), and then using the `Process DRIAMS data` notebook in the `data` folder.


## Siamese Network model

The Siamese Network contains two identical subnetworks with shared weights that work in tandem on two input vectors composed of the MALDI-TOF mass spectra and the chemical fingerprints. The model is trained to minimize the difference between the actual and predicted similarity between pairs of observations

![siamese network](images/Siamese.png "Architecture of the Siamese Network")

The code for the Siamese Netowork model is found in `dd_experiments/SiameseNetworks.py`


## ResMLP model

The ResMLP model includes two projections layers that bring the representations of the MALDI-TOF spectrum and the chemical fingerprint to the same latent dimensionality, before concatenating these vectors and propagating them through a feed-forward neural network with skip-connections.


![siamese network](images/ResMLP.png "Architecture of the ResMLP model")

### Environment Setup

To train ResMLP classifier models with the same setup presented in the paper, we recommend the creation of a suitable Python 3.9 eniromnent with the tool of your choice (e.g. Conda or virtualenv). An example with Conda:

```
conda create -n multimodalAMR python=3.9
conda activate multimodalAMR
```

Clone the repository, and from within the MultimodalAMR folder, nstall the necessary dependencies listed in the requirements.txt file

```
pip install -r requirements.txt
```

Install the multimodal_amr package to access the necessary utilities (from within the MultimodalAMR directory)
```
pip install .
```

### Preprocessing the data

Download the original files from Dryad, at [https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q).
Use the `data/Process Driams data.ipynb` notebook to obtain the preprocessed files containing the MALDI-TOF spectra.



### Training a ResMLP model

To train a ResMLP classifier model with the same configuration as the one presented in the paper, the command used is:

```
python3 gv_experiments/training_scripts/train_ResAMR_classifier.py --experiment_name "myExperiment" --experiment_group "ResMLP" --driams_dataset "B" --seed 0 --split_type "random" --driams_long_table "processed_data/DRIAMS_combined_long_table.csv"  --drugs_df "processed_data/drug_fingerprints.csv" --spectra_matrix "data/DRIAMS-B/spectra_binned_6000_2018.npy" --n_epochs 500 --learning_rate 0.0003 --drug_emb_type "fingerprint" --fingerprint_class "morgan_1024" --fingerprint_size 1024 --patience 50 --batch_size 128
```

### Train a ResMLP model for the single species-single drug comparison

To train a model for the single species-single drug comparison, we first pretrain a ResMLP model on the DRIAMS A dataset except for the samples corresponding to the target (species-drug) combination.

For example, for the target combination ("Staphylococcus aureus", "Oxacillin"), to predict on the split number 0:

```
python3 gv_experiments/training_scripts/pretrain_ResAMR_baseline_comparison.py \
        --experiment_name "PretrainingResMLP" --experiment_group "ResAMR_SingleSpecies_SingleDrug" \
        --seed 42 --driams_long_table "processed_data/DRIAMS_combined_long_table.csv" --splits_file "data/AMR_baseline_splits_noHospitalHygene.json" \
        --spectra_matrix "data/DRIAMS-A/spectra_binned_6000_all.npy" --drugs_df "processed_data/drug_fingerprints.csv" \
        --n_epochs 100 --learning_rate 0.0003 --fingerprint_class "morgan_1024" \
        --patience 50 --driams_dataset "A" --batch_size 128 --root_folder "/home/user/results_folder" \
        --target_drug "Oxacillin" --target_species "Staphylococcus aureus"
```

The resulting model will be saved in the folder `/home/user/results_folder`. For the finetuning, we load the target splits, and tune the model on the target combination data not included in the target split.
For each drug-species combination, we have selected 5 test splits. The finetuning command for split `$split_idx` (which should be a value in [0 1 2 3 4]) is:

```
python3 gv_experiments/training_scripts/finetune_ResAMR_baseline_comparison.py \
        --experiment_name "FinetuningResMLP" --experiment_group "ResAMR_SingleSpecies_SingleDrug" \
        --seed 42 --driams_long_table "processed_data/DRIAMS_combined_long_table.csv" --splits_file "data/AMR_baseline_splits_noHospitalHygene.json" \
        --spectra_matrix "data/DRIAMS-A/spectra_binned_6000_all.npy" --drugs_df "processed_data/drug_fingerprints.csv" \
        --workstations_mapping "data/workstations_mapping.json" \
        --n_epochs 100 --learning_rate 0.0001 --fingerprint_class "morgan_1024" \
        --patience 30 --driams_dataset "A" --batch_size 32 --training_setup $1 \
        --root_folder /fast/gvisona/AMR_Pred \
        --pretrained_checkpoints_folder /home/user/results_folder/outputs/ResAMR_SingleSpecies_SingleDrug/PretrainingResMLP/42/checkpoints
```




The analogous scripts tagged `_filterSpectra` perform the pretraining and finetuning after removing the spectra present in the target split from the pretraining set. In this case, a separate model has to be pretrained for each split, rather than for each drug-species combination as before. The respective commands are


```
python3 gv_experiments/training_scripts/pretrain_ResAMR_baseline_comparison_filterSpectra.py \
        --experiment_name "PretrainingResMLP_filterSpectra" --experiment_group "ResAMR_SingleSpecies_SingleDrug" \
        --seed 42 --driams_long_table "processed_data/DRIAMS_combined_long_table.csv" --splits_file "data/AMR_baseline_splits_noHospitalHygene.json" \
        --spectra_matrix "data/DRIAMS-A/spectra_binned_6000_all.npy" --drugs_df "processed_data/drug_fingerprints.csv" \
        --n_epochs 100 --learning_rate 0.0003 --fingerprint_class "morgan_1024" \
        --patience 50 --driams_dataset "A" --batch_size 128 --root_folder "/home/user/results_folder" \
        --target_drug "Oxacillin" --target_species "Staphylococcus aureus" --split_idx 0
```

for the pretraining, and 

```
python3 gv_experiments/training_scripts/finetune_ResAMR_baseline_comparison_filterSpectra.py \
        --experiment_name "FinetuningResMLP_filterSpectra" --experiment_group "ResAMR_SingleSpecies_SingleDrug" \
        --seed 42 --driams_long_table "processed_data/DRIAMS_combined_long_table.csv" --splits_file "data/AMR_baseline_splits_noHospitalHygene.json" \
        --spectra_matrix "data/DRIAMS-A/spectra_binned_6000_all.npy" --drugs_df "processed_data/drug_fingerprints.csv" \
        --n_epochs 200 --learning_rate 0.0001 --fingerprint_class "morgan_1024" \
        --patience 30 --driams_dataset "A" --batch_size 32 \
        --root_folder /fast/gvisona/AMR_Pred \
        --pretrained_checkpoints_folder /home/user/results_folder/outputs/ResAMR_SingleSpecies_SingleDrug/PretrainingResMLP_filterSpectra \
        --target_drug "Oxacillin" --target_species "Staphylococcus aureus" --split_idx 0
```

