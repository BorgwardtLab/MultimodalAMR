import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.sample_emb = ResMLP(
            config["sample_hidden_layers"],
            config["conv_out_size"] + config["species_embedding_dim"],
            config["sample_embedding_dim"],
            p_dropout=0.2,
        )

        # Maldi-tof spectrum embedding
        self.spectrum_emb = Conv1d_Block(output_dim=config["conv_out_size"])

        # Drugs layers
        if config["drug_emb_type"] == "vae_embedding":
            self.drug_emb = nn.Identity()
        elif config["drug_emb_type"] == "fingerprint":
            self.drug_emb = nn.Sequential(
                nn.Linear(config["fingerprint_size"], config["drug_hidden_dim"]),
                ResMLP(
                    config["drug_hidden_layers"],
                    config["drug_hidden_dim"],
                    config["drug_embedding_dim"],
                    p_dropout=0.2,
                ),
            )

        # Output layer/operation
        # self.out = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.out = nn.Linear(
            config["sample_embedding_dim"] + config["drug_embedding_dim"], 1
        )

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
        # return torch.logit(torch.square(self.out(dr_emb, sample_emb))).view(-1,1)
        return self.out(torch.cat([dr_emb, sample_emb], dim=1))


class AMR_Classifier_Cosine(AMR_Classifier):
    """Overall model definition."""

    def __init__(self, config):
        super().__init__(config)
        # Output layer/operation
        self.out = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.out = nn.Linear(config["sample_embedding_dim"]+config["drug_embedding_dim"], 1)

    def forward(self, batch):
        species_idx, x_spectrum, dr_tensor, response, dataset = batch
        spectrum_embedding = self.embed_spectrum(x_spectrum)
        sample_emb = self.embed_sample(species_idx, spectrum_embedding)
        dr_emb = self.drug_emb(dr_tensor)
        return torch.logit(torch.square(self.out(dr_emb, sample_emb))).view(-1, 1)


class MLP_Classifier(nn.Module):
    """Overall model definition."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.projection_layer = nn.Sequential(
            nn.Linear(config["input_size"], config["hidden_size"]), nn.ReLU()
        )
        self.net = ResMLP(
            config["n_hidden_layers"], config["hidden_size"], 1, p_dropout=0.2
        )

    def forward(self, batch):
        species_idx, x_spectrum, dr_tensor, response, dataset = batch
        return self.net(
            self.projection_layer(torch.cat([x_spectrum, dr_tensor], dim=1))
        )


class SpeciesBaseline_MLP_Classifier(MLP_Classifier):
    """Overall model definition."""

    def __init__(self, config):
        super().__init__(config)
        self.n_species = config["n_unique_species"]
        # self.config = config

        # self.projection_layer = nn.Sequential(nn.Linear(config["input_size"], config["hidden_size"]), nn.ReLU())
        # self.net = ResMLP(config["n_hidden_layers"],
        #                          config["hidden_size"],
        #                          1, p_dropout=0.2)

    def forward(self, batch):
        species_idx, x_spectrum, dr_tensor, response, dataset = batch
        species_tensor = F.one_hot(species_idx, num_classes=self.n_species).squeeze()
        return self.net(
            self.projection_layer(torch.cat([species_tensor, dr_tensor], dim=1))
        )


# class AMR_Classifier_noSP(nn.Module):
#     """Overall model definition."""

#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         self.sample_emb = ResMLP(
#             config["sample_hidden_layers"],
#             config["conv_out_size"],
#             config["sample_embedding_dim"],
#             p_dropout=0.2,
#         )

#         # Maldi-tof spectrum embedding
#         self.spectrum_emb = Conv1d_Block(output_dim=config["conv_out_size"])

#         # Drugs layers
#         if config["drug_emb_type"] == "vae_embedding":
#             self.drug_emb = nn.Identity()
#         elif config["drug_emb_type"] == "fingerprint":
#             self.drug_emb = nn.Sequential(
#                 nn.Linear(config["fingerprint_size"], config["drug_hidden_dim"]),
#                 ResMLP(
#                     config["drug_hidden_layers"],
#                     config["drug_hidden_dim"],
#                     config["drug_embedding_dim"],
#                     p_dropout=0.2,
#                 ),
#             )

#         # Output layer/operation
#         # self.out = nn.CosineSimilarity(dim=1, eps=1e-6)
#         self.out = nn.Linear(
#             config["sample_embedding_dim"] + config["drug_embedding_dim"], 1
#         )

#     def embed_spectrum(self, x_spectrum):
#         return self.spectrum_emb(torch.unsqueeze(x_spectrum, dim=1))

#     def forward(self, batch):
#         species_idx, x_spectrum, dr_tensor, response, dataset = batch
#         spectrum_embedding = self.embed_spectrum(x_spectrum)
#         sample_emb = self.sample_emb(spectrum_embedding)
#         dr_emb = self.drug_emb(dr_tensor)
#         return self.out(torch.cat([dr_emb, sample_emb], dim=1))


class Residual_AMR_Classifier(nn.Module):
    """Overall model definition."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Species embedding
        if config["species_embedding_dim"]>0:
            self.species_emd = nn.Embedding(
                config["n_unique_species"], config["species_embedding_dim"]
            )
        else:
            self.species_emd = None

        # Sample embedding
        if config["species_embedding_dim"]==0 and config["conv_out_size"]==config["sample_embedding_dim"]:
            self.sample_emb = nn.Identity()
        elif config["species_embedding_dim"]==0:
            self.sample_emb = nn.Linear(
                config["conv_out_size"],
                config["sample_embedding_dim"],
            )
        else:
            self.sample_emb = nn.Linear(
                config["conv_out_size"] + config["species_embedding_dim"],
                config["sample_embedding_dim"],
            )

        # Maldi-tof spectrum embedding
        # self.spectrum_emb = Conv1d_Block(output_dim=config["conv_out_size"])
        self.spectrum_emb = nn.Linear(6000 ,config["conv_out_size"])

        # Drugs layers
        if config["drug_emb_type"] == "vae_embedding":
            self.drug_emb = nn.Identity()
        elif config["drug_emb_type"] == "fingerprint":
            self.drug_emb = nn.Linear(
                config["fingerprint_size"], config["drug_embedding_dim"]
            )

        # Output network
        self.net = ResMLP(
            config["n_hidden_layers"],
            config["sample_embedding_dim"] + config["drug_embedding_dim"],
            1,
            p_dropout=0.2,
        )

    def embed_spectrum(self, x_spectrum):
        # return self.spectrum_emb(torch.unsqueeze(x_spectrum, dim=1))
        return self.spectrum_emb(x_spectrum)

    def embed_sample(self, species_idx, spectrum_embedding):
        if self.species_emd is not None:
            species_embedding = self.species_emd(species_idx.view(-1))
            cat_emb = torch.cat([species_embedding, spectrum_embedding], dim=1)
            return self.sample_emb(cat_emb)
        else:
            return self.sample_emb(spectrum_embedding)

    def forward(self, batch):
        species_idx, x_spectrum, dr_tensor, response, dataset = batch
        spectrum_embedding = self.embed_spectrum(x_spectrum)
        sample_emb = self.embed_sample(species_idx, spectrum_embedding)
        dr_emb = self.drug_emb(dr_tensor)
        return self.net(torch.cat([dr_emb, sample_emb], dim=1))



class SpeciesBaseline_ResAMR_Classifier(Residual_AMR_Classifier):
    """Overall model definition."""

    def __init__(self, config):
        super().__init__(config)
        self.n_species = config["n_unique_species"]
        self.spectrum_emb = nn.Linear(self.n_species ,config["conv_out_size"])


    def forward(self, batch):
        species_idx, x_spectrum, dr_tensor, response, dataset = batch
        species_tensor = F.one_hot(species_idx, num_classes=self.n_species).squeeze().float()
        spectrum_embedding = self.embed_spectrum(species_tensor)
        sample_emb = self.embed_sample(species_idx, spectrum_embedding)
        dr_emb = self.drug_emb(dr_tensor)
        return self.net(torch.cat([dr_emb, sample_emb], dim=1))