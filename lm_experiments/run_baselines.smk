"""

Snakefile for data and imputation.
Execution: sbatch snakemake
Plot DAG: snakemake --snakefile deepof_experiments.smk --forceall --dag | dot -Tpdf > deepof_experiments_DAG.pdf
Plot rule graph: snakemake --snakefile deepof_experiments.smk --forceall --rulegraph | dot -Tpdf > deepof_experiments_RULEGRAPH.pdf

"""

outpath = ""

model = ["LogisticRegression", "GBM", "MLP"]
dataset = ["A", "B", "C", "D"]


rule baselines:
    input:
        # Train a variety of models
        expand(
            outpath + "DRIAMS_{dataset}/{model}/config.json",
            dataset=dataset,
            model=model,
        ),


rule train_baselines:
    output:
        trained_models=os.path.join(outpath, "DRIAMS_{dataset}/{model}/config.json"),
    shell:
        "python DRIAMS_baselines_eval.py "
        "--dataset {wildcards.dataset} "
        "--model {wildcards.model} "
        "--spectra_matrix_path ../data/DRIAMS-{wildcards.dataset}/spectra_binned_6000_reprocessed.csv "
        "--output . "
        "--threshold 100 "
        "--n_random_iter 10 "
        "--min_samples_per_class 50"
