import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm

if __name__ == "__main__":

    # Process raw data files into spectral dataframes

    long_table = pd.read_csv("../processed_data/DRIAMS_combined_long_table.csv")

    for dset in ["A", "B", "C", "D"]:

        for year in ["2015", "2016", "2017", "2018"]:

            print("Processing DRIAMS-{}".format(dset))

            full_samples = long_table.loc[long_table.dataset == dset].sample_id.unique()

            #        current_samples = sorted(list(long_table["sample_id"].unique()))
            samples_spectra = []

            current_samples = []
            for i, sample_id in tqdm(enumerate(full_samples)):

                try:
                    spectrum = pd.read_csv(
                        f"../data/DRIAMS-{dset}/binned_6000/{year}/{sample_id}.txt",
                        sep=" ",
                        index_col=0,
                    )
                    samples_spectra.append(spectrum.values.flatten())
                    current_samples.append(sample_id)

                except FileNotFoundError:
                    continue

            if len(samples_spectra) > 0:

                samples_spectra = np.vstack(samples_spectra)

                spectra_df = pd.DataFrame(data=samples_spectra, index=current_samples)

                spectra_df.to_csv(
                    "../data/DRIAMS-{}/spectra_binned_6000_{}_reprocessed.csv".format(
                        dset, year
                    )
                )

        print("DRIAMS-{} processed!".format(dset))
