import torch
import torch.nn as nn
from .modules import MLP_Block, Conv1d_Block, ResMLP



class AMR_Classifier(nn.Module):
    """Overall model definition."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Species embedding
        self.species_emd = nn.Embedding(
            config["n_unique_species"], config["species_embedding_dim"]
        )

        # Sample embedding
        # sample_embedder_layers_sizes = (
        #     [
        #         config["conv_out_size"]
        #         + config["species_embedding_dim"]
        #     ]
        #     + config["sample_hidden_layers"]
        #     + [config["sample_embedding_dim"]]
        # )
        # self.sample_emb = MLP_Block(sample_embedder_layers_sizes)
        self.sample_emb = ResMLP(config["sample_hidden_layers"], 
                                 config["conv_out_size"] + config["species_embedding_dim"],
                                 config["sample_embedding_dim"], p_dropout=0.2)


        # Maldi-tof spectrum embedding
        self.spectrum_emb = Conv1d_Block(output_dim=config["conv_out_size"])

        # Drugs layers
        if config["drug_emb_type"]=="vae_embedding":
            self.drug_emb = nn.Identity()
        elif config["drug_emb_type"]=="fingerprint":
            self.drug_emb = nn.Sequential(nn.Linear(config["fingerprint_size"], config["drug_hidden_dim"]),
                                         ResMLP(config["drug_hidden_layers"], config["drug_hidden_dim"], config["drug_embedding_dim"], p_dropout=0.2))

        # Output layer/operation
        # self.out = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.out = nn.Linear(config["sample_embedding_dim"]+config["drug_embedding_dim"], 1)




    def embed_spectrum(self, x_spectrum):
        return self.spectrum_emb(torch.unsqueeze(x_spectrum, dim=1))

    def embed_sample(self, species_idx, spectrum_embedding):
        species_embedding = self.species_emd(species_idx.view(-1))
        cat_emb = torch.cat([species_embedding, spectrum_embedding], dim=1)
        return self.sample_emb(cat_emb)


    def forward(self, batch):
        species_idx, x_spectrum, dr_tensor, response, dataset = batch
        spectrum_embedding = self.embed_spectrum(x_spectrum)
        sample_emb = self.embed_sample(species_idx, spectrum_embedding)
        dr_emb = self.drug_emb(dr_tensor)
        return self.out(torch.cat([dr_emb, sample_emb], dim=1))