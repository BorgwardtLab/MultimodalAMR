import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm

if __name__ == "__main__":

    # Process raw data files into spectral dataframes

    long_table = pd.read_csv("../processed_data/DRIAMS_combined_long_table.csv")

    for dataset in ["A", "B", "C", "D"]:

        print("Processing DRIAMS-{}".format(dataset))

        current_samples = long_table[long_table["dataset"] == dataset].reset_index()

        samples_spectra = []

        for i, sample_id in tqdm(enumerate(all_samples)):
            spectrum = pd.read_csv(
                f"../data/DRIAMS-B/binned_6000/2018/{sample_id}.txt",
                sep=" ",
                index_col=0,
            )
            samples_spectra.append(spectrum.values.flatten())
        samples_spectra = np.vstack(samples_spectra)

        spectra_df = pd.DataFrame(data=samples_spectra, index=all_samples)

        spectra_df.to_csv(
            "../data/DRIAMS-{}/spectra_binned_6000_2018_reprocessed.csv".format(dataset)
        )

        print("DRIAMS-{} processed!".format(dataset))
