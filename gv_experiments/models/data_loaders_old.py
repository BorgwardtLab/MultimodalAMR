import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict





def train_valid_test_split(input_df, valid_size=0.2, test_size=0.2, zero_shot=False, random_state=42):
    ''' Train - validation - test split stratified by species and response. 
    As some species - response combinations might appear only once in the dataset, 
    we include all these in the training set. 
    '''
    df = input_df.copy()
    # aggreagate species and response for easier indexing
    df['species_w_response'] = df[['species', 'response']].astype(str).agg('-'.join, axis=1)
    
    # filter out categories with a single entry
    single_entry_idx = [gr for gr, df_gr in df.groupby(['species_w_response']) if len(df_gr) == 1]
    
    df_single = df[df['species_w_response'].isin(single_entry_idx)]
    df_filtered = df[~df['species_w_response'].isin(single_entry_idx)]
    
    # get stratification variable
    y_str = LabelEncoder().fit_transform(df_filtered['species_w_response'])

    # train + validation and test split
    remaining_idx, test_idx = train_test_split(
        np.arange(len(y_str)),
        test_size=test_size,
        stratify=y_str,
        random_state=random_state
    )

    df_single = df_single.drop("species_w_response", axis=1)
    df_filtered = df_filtered.drop("species_w_response", axis=1)

    if not zero_shot:
        # train and validation split
        train_idx, valid_idx = train_test_split(
            np.arange(len(y_str[remaining_idx])),
            test_size=valid_size,
            stratify=y_str[remaining_idx],
            random_state=random_state
        )

        return pd.concat([df_single, df_filtered.iloc[train_idx,:]], axis=0), df_filtered.iloc[valid_idx, :], df_filtered.iloc[test_idx, :]
    else:
        return pd.concat([df_single, df_filtered.iloc[remaining_idx,:]], axis=0), df_filtered.iloc[test_idx, :]
    

def zero_shot_train_valid_test_split(df, drugs, valid_size=0.2, random_state=42):
    ''' Train - validation - test split for zero-shot learning.
    '''
    # select for testing based on list of drugs
    df_test = df[df['drug'].isin(drugs)]
    
    # stratified train - validation split based on species and response
    df_train, df_valid = train_valid_test_split(df[~df['drug'].isin(drugs)].copy(), valid_size=valid_size, random_state=random_state, zero_shot=True)
    
    return df_train, df_valid, df_test


class DrugResistanceDataset(Dataset):
    def __init__(
        self,
        long_table,
        spectra_matrix,
        phylogen_df,
        drugs_fingerprints,
        fingerprint_class="MACCS",
        split=None,
        random_state=42,
    ):
        """
        Dataset class to retrieve combinations of species-samples-drugs-drug resistance quadruplets

        :long_table: DataFrame from processed csv file with the quadruplets for each hospital
        :spectra_matrix: matrix of 6000-dimensional MALDI-TOF spectra
        :drug_fingerprints: DataFrame of chemical fingerprints
        :phylogen_matrix: ?
        :fingerprint_class: which chemical fingerprint to use. Choice of 'MACCS', 'morgan', or 'pubchem'
        """
        long_table = long_table[long_table["species"].isin(phylogen_df.index)]

        self.long_table = long_table
        self.spectra_tensor = torch.tensor(spectra_matrix).float()

        self.idx2phylo_species = {i: d for i, d in enumerate(phylogen_df.index)}
        self.phylo_species2idx = {d: i for i,
                                  d in self.idx2phylo_species.items()}
        self.phylo_embeddings_tensor = torch.tensor(phylogen_df.values).float()

        sorted_species = sorted(long_table["species"].unique())
        self.idx2species = {i: s for i, s in enumerate(sorted_species)}
        self.species2idx = {s: i for i, s in self.idx2species.items()}

        sorted_samples = sorted(long_table["sample_id"].unique())
        self.idx2sample = {i: smp for i, smp in enumerate(sorted_samples)}
        self.sample2idx = {smp: i for i, smp in self.idx2sample.items()}

        self.idx2drug = {i: d for i,
                         d in drugs_fingerprints["drug"].iteritems()}
        self.drug2idx = {d: i for i, d in self.idx2drug.items()}

        self.drugs_tensor = torch.tensor(
            [
                [int(v) for v in list(row[fingerprint_class + "_fp"])]
                for i, row in drugs_fingerprints.iterrows()
            ]
        ).float()

        if split is not None:
            train_table, val_table, test_table = train_valid_test_split(
                self.long_table, random_state=random_state)
            if split == "training":
                self.long_table = train_table
            elif split == "validation":
                self.long_table = val_table
            elif split == "test":
                self.long_table = test_table
            else:
                raise ValueError(
                    "Invalid split selection, choose one of training, validation or test."
                )

    def __len__(self):
        return len(self.long_table)

    def __getitem__(self, idx):
        species, sample_id, drug_name, response = self.long_table.iloc[idx]

        fprint_tensor = self.drugs_tensor[self.drug2idx[drug_name]]
        response = torch.tensor(response).float()

        spectrum = self.spectra_tensor[self.sample2idx[sample_id], :]
        phylo_species_embedding = self.phylo_embeddings_tensor[
            self.phylo_species2idx[species], :
        ]

        species_idx = torch.LongTensor([self.species2idx[species]])
        return species_idx, phylo_species_embedding, spectrum, fprint_tensor, response


class SampleEmbDataset(Dataset):
    def __init__(self, long_table, spectra_matrix, phylogen_df):
        long_table = long_table[long_table["species"].isin(phylogen_df.index)]
        self.spectra_tensor = torch.tensor(spectra_matrix).float()

        self.idx2phylo_species = {i: d for i, d in enumerate(phylogen_df.index)}
        self.phylo_species2idx = {d: i for i,
                                  d in self.idx2phylo_species.items()}
        self.phylo_embeddings_tensor = torch.tensor(phylogen_df.values).float()

        sorted_species = sorted(long_table["species"].unique())
        self.idx2species = {i: s for i, s in enumerate(sorted_species)}
        self.species2idx = {s: i for i, s in self.idx2species.items()}

        sorted_samples = sorted(long_table["sample_id"].unique())
        self.idx2sample = {i: smp for i, smp in enumerate(sorted_samples)}
        self.sample2idx = {smp: i for i, smp in self.idx2sample.items()}

        # Filter long table
        long_table = long_table.drop(["drug", "response"], axis=1)
        long_table = long_table.drop_duplicates()
        self.long_table = long_table

    def __len__(self):
        return len(self.long_table)

    def __getitem__(self, idx):
        species, sample_id = self.long_table.iloc[idx]
        spectrum = self.spectra_tensor[self.sample2idx[sample_id], :]
        phylo_species_embedding = self.phylo_embeddings_tensor[
            self.phylo_species2idx[species], :
        ]

        species_idx = torch.LongTensor([self.species2idx[species]])
        return species_idx, phylo_species_embedding, spectrum




class DrugEmbDataset(Dataset):
    def __init__(
            self,
            drugs_fingerprints,
            fingerprint_class="MACCS",
        ):

        self.idx2drug = {i: d for i,
                         d in drugs_fingerprints["drug"].iteritems()}
        self.drug2idx = {d: i for i, d in self.idx2drug.items()}

        self.drugs_tensor = torch.tensor(
            [
                [int(v) for v in list(row[fingerprint_class + "_fp"])]
                for i, row in drugs_fingerprints.iterrows()
            ]
        ).float()

    def __len__(self):
        return len(self.idx2drug)
    
    def __getitem__(self, idx):
        fprint_tensor = self.drugs_tensor[idx]
        return fprint_tensor



class ZeroShotDrugDataset(DrugResistanceDataset):
    def __init__(
        self,
        long_table,
        spectra_matrix,
        phylogen_df,
        drugs_fingerprints,
        fingerprint_class="MACCS",
        zs_drugs=["Ciprofloxacin"],
        split=None,
        random_state=42,
    ):
        """
        Dataset class to retrieve combinations of species-samples-drugs-drug resistance quadruplets

        :long_table: DataFrame from processed csv file with the quadruplets for each hospital
        :spectra_matrix: matrix of 6000-dimensional MALDI-TOF spectra
        :drug_fingerprints: DataFrame of chemical fingerprints
        :phylogen_matrix: ?
        :fingerprint_class: which chemical fingerprint to use. Choice of 'MACCS', 'morgan', or 'pubchem'
        """
        long_table = long_table[long_table["species"].isin(phylogen_df.index)]

        self.long_table = long_table
        self.spectra_tensor = torch.tensor(spectra_matrix).float()

        self.idx2phylo_species = {i: d for i, d in enumerate(phylogen_df.index)}
        self.phylo_species2idx = {d: i for i,
                                  d in self.idx2phylo_species.items()}
        self.phylo_embeddings_tensor = torch.tensor(phylogen_df.values).float()

        sorted_species = sorted(long_table["species"].unique())
        self.idx2species = {i: s for i, s in enumerate(sorted_species)}
        self.species2idx = {s: i for i, s in self.idx2species.items()}

        sorted_samples = sorted(long_table["sample_id"].unique())
        self.idx2sample = {i: smp for i, smp in enumerate(sorted_samples)}
        self.sample2idx = {smp: i for i, smp in self.idx2sample.items()}

        self.idx2drug = {i: d for i,
                         d in drugs_fingerprints["drug"].iteritems()}
        self.drug2idx = {d: i for i, d in self.idx2drug.items()}

        self.drugs_tensor = torch.tensor(
            [
                [int(v) for v in list(row[fingerprint_class + "_fp"])]
                for i, row in drugs_fingerprints.iterrows()
            ]
        ).float()

        if split is not None:
            train_table, val_table, test_table = zero_shot_train_valid_test_split(
                self.long_table, zs_drugs, random_state=random_state)
            if split == "training":
                self.long_table = train_table
            elif split == "validation":
                self.long_table = val_table
            elif split == "test":
                self.long_table = test_table
            else:
                raise ValueError(
                    "Invalid split selection, choose one of training, validation or test."
                )