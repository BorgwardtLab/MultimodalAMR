#!/bin/bash
#SBATCH --ntasks=1 #each job has one task
#SBATCH --cpus-per-task=1 # each task uses 1 cpu
#SBATCH --partition=urtgen_24hrs
#SBATCH --mem-per-cpu=100000 

#Parameters
dir=/home/
longTable=/home/DRIAMS_combined_long_table.csv
driam=B
path=/home/
export longTable driam path

cd  $dir
#mkdir $path

module load R/3.5.1
module load EasyBuild
module load Python/3.7.4-GCCcore-8.3.0
python

############
# Workflow #
############

#Create train - val (all drugs and all species are represented in the training data, but that no specific drug-species pair appears in more than one set) - test (zero shot)
# and random_train - random_test
Rscript createInputData_baseline.R "$driam" "$path"
Rscript createInputData_siamese.R "$driam" "$path"

#"baseline species" set-up
Rscript baseline_species.R

#"random baseline" and "spectrum similarity" set-ups
time python similarity_baseline.py
time python similarity_baseline_truncated.py

#Siamese networks
time python SiameseNetworks.py
Rscript siamese_outputFormat.R
Rscript modelAndEvaluation.R
