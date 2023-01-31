# AMR_Pred

Core repository for the RECOMB 2023 submission _Joint proteomics and chemical space modelling for antimicrobial resistance prediction_ from Giovanni Visona, Yiran Li, Diane Duroux, Lucas Miranda, Emese Sukei, Karsten Borgwardt, and Carlos Oliver.



## Training a ResAMR model

To train a ResAMR classifier model with the same configuration as the one presented in the paper, the command used (from the gv_experiments folder) is:

```
python3 train_ResAMR_classifier.py --experiment_name "noCNN_emb512" --experiment_group "ResAMR" \
         --training_setup $1 --driams_long_table "processed_data/DRIAMS_combined_long_table.csv" \
         --drugs_df "processed_data/drug_fingerprints.csv" \
         --n_epochs 500 --learning_rate 0.0003 --fingerprint_class "morgan_1024" --patience 50 --batch_size 128 \
```