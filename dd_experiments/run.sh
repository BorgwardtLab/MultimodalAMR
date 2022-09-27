#!/bin/bash
#SBATCH --ntasks=1 #each job has one task
#SBATCH --cpus-per-task=1 # each task uses 1 cpu
#SBATCH --partition=urtgen_24hrs
#SBATCH --mem-per-cpu=100000 

#Parameters
dir=/massstorage/URT/GEN/BIO3/PRIV/Team/Diane/RESEARCH/Hackathon/October/DRIAMSB
longTable=/home/gallia/scratch/u230399/DRIAMS_combined_long_table.csv
fingerprints=/home/gallia/scratch/u230399/drug_fingerprints.csv
driam=B
drug=Cefepime
#species="Staphylococcus epidermidis"
path=/home/gallia/scratch/u230399/DRIAMS-$driam/$drug/
export longTable fingerprints driam drug species path

cd  $dir
mkdir $path

#Load R and python
module load R/3.5.1
module load EasyBuild
module load Python/3.7.4-GCCcore-8.3.0
python

############
# Workflow #
############

#Select ids for data splitting
time python data_utils.py

#Create train - val (all drugs and all species are represented in the training data, but that no specific drug-species pair appears in more than one set) - test (zero shot)
Rscript createInputData.R "$driam" "$drug" "$path"

#Siamese networks - generate embeddings
time python SiameseNetworks.py

#Model and evaluation
Rscript modelAndEvaluation.R "$driam" "$drug"

