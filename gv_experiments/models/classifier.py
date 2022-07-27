import torch
import torch.nn as nn



class MLP_Block(nn.Module):
    """Embedding block for the chemical information."""

    def __init__(self, embedder_layers_sizes, dropout=0.1):
        super().__init__()

        # Fully-connected layers + ReLU
        embedder_layers = []
        for k in range(len(embedder_layers_sizes) - 1):
            embedder_layers.append(
                nn.Linear(embedder_layers_sizes[k], embedder_layers_sizes[k + 1])
            )
            # if k<len(embedder_layers_sizes) - 2:
            embedder_layers.extend([nn.ReLU(), nn.Dropout(dropout)])

        self.embedder = nn.Sequential(*embedder_layers)

    def forward(self, x):
        return self.embedder(x)


class SpectrumEmbedding(nn.Module):
    """CNN-based embedding block for the Maldi-tof spectra."""

    def __init__(self,
                #  in_dim=1,
                 kernel_sizes=[5, 7, 11, 13],
                 num_kernels=[32, 16, 8, 8]):
        super().__init__()

        # Convolutional layers + ReLU + Dropout + MaxPool
        spectrum_embedder_layers = []
        num_kernels = [1] + num_kernels
        for i in range(len(num_kernels)-1):
            spectrum_embedder_layers.append(nn.Conv1d(in_channels=num_kernels[i],
                                                      out_channels=num_kernels[i+1],
                                                      kernel_size=kernel_sizes[i],
                                                      stride=3))
            spectrum_embedder_layers.extend(
                [nn.BatchNorm1d(num_kernels[i+1]), nn.ReLU()])

        self.spectrum_embedder = nn.Sequential(*spectrum_embedder_layers)

        self.out_layer = nn.Sequential(nn.Flatten(), nn.Linear(552, 64))

    def forward(self, x):
        # `x` must have input shape matching the requirement of `nn.Conv1d`: (N, C_in, L_in)
        cnn_out = self.spectrum_embedder(x)

        return self.out_layer(cnn_out)



class SharedEmbeddingClassifier(nn.Module):
    """Overall model definition."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # self.drug_emb = MLP_Block(drug_embedder_layers_sizes)

        # Species embedding
        self.species_emd = nn.Embedding(
            config["n_unique_species"], config["species_embedding_dim"]
        )

        # # Phylogenetic tree embedding
        sample_embedder_layers_sizes = (
            [
                config["conv_out_size"]
                + config["species_embedding_dim"]
            ]
            + config["sample_hidden_layers"]
            + [config["sample_embedding_dim"]]
        )

        self.sample_emb = MLP_Block(sample_embedder_layers_sizes)

        # Maldi-tof spectrum embedding
        self.spectrum_emb = SpectrumEmbedding()

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
        species_idx, x_spectrum, dr_emb, _ = batch
        spectrum_embedding = self.embed_spectrum(x_spectrum)
        sample_emb = self.embed_sample(species_idx, spectrum_embedding)
        return self.out(torch.cat([dr_emb, sample_emb], dim=1))