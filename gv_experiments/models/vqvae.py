import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# class VectorQuantization(Function):
#     @staticmethod
#     def forward(ctx, inputs, codebook):
#         with torch.no_grad():
#             embedding_size = codebook.size(1)
#             inputs_size = inputs.size()
#             inputs_flatten = inputs.view(-1, embedding_size)

#             codebook_sqr = torch.sum(codebook ** 2, dim=1)
#             inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

#             # Compute the distances to the codebook
#             distances = torch.addmm(codebook_sqr + inputs_sqr,
#                                     inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

#             _, indices_flatten = torch.min(distances, dim=1)
#             indices = indices_flatten.view(*inputs_size[:-1])
#             ctx.mark_non_differentiable(indices)

#             return indices

#     @staticmethod
#     def backward(ctx, grad_output):
#         raise RuntimeError('Trying to call `.grad()` on graph containing '
#                            '`VectorQuantization`. The function `VectorQuantization` '
#                            'is not differentiable. Use `VectorQuantizationStraightThrough` '
#                            'if you want a straight-through estimator of the gradient.')


# class VectorQuantizationStraightThrough(Function):
#     @staticmethod
#     def forward(ctx, inputs, codebook):
#         indices = vq(inputs, codebook)
#         indices_flatten = indices.view(-1)
#         ctx.save_for_backward(indices_flatten, codebook)
#         ctx.mark_non_differentiable(indices_flatten)

#         codes_flatten = torch.index_select(codebook, dim=0,
#                                            index=indices_flatten)
#         codes = codes_flatten.view_as(inputs)

#         return (codes, indices_flatten)

#     @staticmethod
#     def backward(ctx, grad_output, grad_indices):
#         grad_inputs, grad_codebook = None, None

#         if ctx.needs_input_grad[0]:
#             # Straight-through estimator
#             grad_inputs = grad_output.clone()
#         if ctx.needs_input_grad[1]:
#             # Gradient wrt. the codebook
#             indices, codebook = ctx.saved_tensors
#             embedding_size = codebook.size(1)

#             grad_output_flatten = (grad_output.contiguous()
#                                               .view(-1, embedding_size))
#             grad_codebook = torch.zeros_like(codebook)
#             grad_codebook.index_add_(0, indices, grad_output_flatten)

#         return (grad_inputs, grad_codebook)


# vq = VectorQuantization.apply
# vq_st = VectorQuantizationStraightThrough.apply


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class ViewLayer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, emb_dim, K=512, comb_size=4, committment_cost=0.25):
        super().__init__()
        self.comb_size = comb_size
        self.code_dim = emb_dim // comb_size
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            ResBlock(emb_dim),
            ResBlock(emb_dim),
            ViewLayer([-1, self.comb_size, self.code_dim]),
        )

        self.codebook = VectorQuantizer(K, self.code_dim, committment_cost)

        self.decoder = nn.Sequential(
            ResBlock(emb_dim),
            ResBlock(emb_dim),
            ResBlock(emb_dim),
            nn.Linear(emb_dim, input_dim),
            nn.Sigmoid(),
        )

        self.apply(weights_init)

    def encode(self, x):
        # batch_size = x.size(0)
        # .view(batch_size, self.comb_size, self.code_dim)
        z = self.encoder(x)
        loss, quantized, perplexity, encodings = self.codebook(z)
        return loss, quantized, perplexity, encodings

    def decode(self, latents):
        batch_size = latents.size(0)
        z_q_x = self.codebook.embedding(latents).view(
            batch_size, -1
        )  # .permute(0, 3, 1, 2) (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        batch_size = x.size(0)
        # z = self.encoder(x)
        # loss, quantized, perplexity, _ = self.codebook(z)
        loss, quantized, perplexity, encodings = self.encode(x)
        x_recon = self.decoder(quantized.view(batch_size, -1))
        return loss, x_recon, perplexity


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings


# class VectorQuantizerEMA(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
#         super(VectorQuantizerEMA, self).__init__()

#         self._embedding_dim = embedding_dim
#         self._num_embeddings = num_embeddings

#         self._embedding = nn.Embedding(
#             self._num_embeddings, self._embedding_dim)
#         self._embedding.weight.data.normal_()
#         self._commitment_cost = commitment_cost

#         self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
#         self._ema_w = nn.Parameter(torch.Tensor(
#             num_embeddings, self._embedding_dim))
#         self._ema_w.data.normal_()

#         self._decay = decay
#         self._epsilon = epsilon

#     def forward(self, inputs):
#         # convert inputs from BCHW -> BHWC
#         # inputs = inputs.permute(0, 2, 3, 1).contiguous()
#         input_shape = inputs.shape

#         # Flatten input
#         flat_input = inputs.view(-1, self._embedding_dim)

#         # Calculate distances
#         distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
#                      + torch.sum(self._embedding.weight**2, dim=1)
#                      - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

#         # Encoding
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
#         encodings = torch.zeros(
#             encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
#         encodings.scatter_(1, encoding_indices, 1)

#         # Quantize and unflatten
#         quantized = torch.matmul(
#             encodings, self._embedding.weight).view(input_shape)

#         # Use EMA to update the embedding vectors
#         if self.training:
#             self._ema_cluster_size = self._ema_cluster_size * self._decay + \
#                 (1 - self._decay) * torch.sum(encodings, 0)

#             # Laplace smoothing of the cluster size
#             n = torch.sum(self._ema_cluster_size.data)
#             self._ema_cluster_size = (
#                 (self._ema_cluster_size + self._epsilon)
#                 / (n + self._num_embeddings * self._epsilon) * n)

#             dw = torch.matmul(encodings.t(), flat_input)
#             self._ema_w = nn.Parameter(
#                 self._ema_w * self._decay + (1 - self._decay) * dw)

#             self._embedding.weight = nn.Parameter(
#                 self._ema_w / self._ema_cluster_size.unsqueeze(1))

#         # Loss
#         e_latent_loss = F.mse_loss(quantized.detach(), inputs)
#         loss = self._commitment_cost * e_latent_loss

#         # Straight Through Estimator
#         quantized = inputs + (quantized - inputs).detach()
#         avg_probs = torch.mean(encodings, dim=0)
#         perplexity = torch.exp(-torch.sum(avg_probs *
#                                torch.log(avg_probs + 1e-10)))

#         return loss, quantized, perplexity, encodings


# class VectorQuantizedVAE(nn.Module):
#     def __init__(self, input_dim, emb_dim, K=512, comb_size=4):
#         super().__init__()
#         self.comb_size = comb_size
#         self.code_dim = emb_dim//comb_size
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, emb_dim),
#             nn.BatchNorm1d(emb_dim),
#             nn.ReLU(),
#             nn.Linear(emb_dim, emb_dim),
#             ResBlock(emb_dim),
#             ResBlock(emb_dim),
#             ViewLayer([-1, self.comb_size, self.code_dim])
#         )

#         self.codebook = VQEmbedding(K, self.code_dim)

#         self.decoder = nn.Sequential(
#             ResBlock(emb_dim),
#             ResBlock(emb_dim),
#             ResBlock(emb_dim),
#             nn.Linear(emb_dim, input_dim),
#             nn.Sigmoid()
#         )

#         self.apply(weights_init)

#     def encode(self, x):
#         # batch_size = x.size(0)
#         z_e_x = self.encoder(x)#.view(batch_size, self.comb_size, self.code_dim)
#         latents = self.codebook(z_e_x)
#         return latents

#     def decode(self, latents):
#         batch_size = latents.size(0)
#         z_q_x = self.codebook.embedding(latents).view(batch_size, -1)  #  .permute(0, 3, 1, 2) (B, D, H, W)
#         x_tilde = self.decoder(z_q_x)
#         return x_tilde

#     def forward(self, x):
#         batch_size = x.size(0)
#         z_e_x = self.encoder(x)
#         z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
#         x_tilde = self.decoder(z_q_x_st.view(batch_size, -1))
#         return x_tilde, z_e_x , z_q_x


# class VQEmbedding(nn.Module):
#     def __init__(self, K, D):
#         super().__init__()
#         self.embedding = nn.Embedding(K, D)
#         self.embedding.weight.data.uniform_(-1./K, 1./K)

#     def forward(self, z_e_x):
#         latents = vq(z_e_x, self.embedding.weight)
#         return latents

#     def straight_through(self, z_e_x):
#         z_q_x, indices = vq_st(z_e_x, self.embedding.weight.detach())
#         z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
#             dim=0, index=indices)
#         z_q_x_bar = z_q_x_bar_flatten.view_as(z_q_x)
#         return z_q_x, z_q_x_bar


class ResBlock(nn.Module):
    def __init__(self, dim, p_dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(), nn.Linear(dim, dim), nn.Dropout(p_dropout), nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        return x + self.block(x)
