import torch
import torch.nn as nn
from .utils import calc_conv1d_output_size


class ResBlock(nn.Module):
    def __init__(self, dim, p_dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(), nn.Linear(dim, dim), nn.Dropout(p_dropout), nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class ResMLP(nn.Module):
    def __init__(self, n_layers, dim, output_dim, p_dropout=0.2):
        super().__init__()
        layers = [ResBlock(dim, p_dropout=p_dropout) for _ in range(n_layers)] + [
            nn.Linear(dim, output_dim)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLP_Block(nn.Module):
    def __init__(self, layers_sizes, dropout=0.1):
        super().__init__()

        # Fully-connected layers + ReLU
        layers = []
        for k in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layers_sizes[k], layers_sizes[k + 1]))
            if k < len(layers_sizes) - 2:
                layers.extend([nn.ReLU(), nn.Dropout(dropout)])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Conv1d_Block(nn.Module):
    def __init__(
        self,
        input_dim=6000,
        kernel_sizes=[5, 7, 11, 9],
        num_kernels=[32, 16, 8, 8],
        stride=3,
        padding=1,
        output_dim=64,
    ):
        super().__init__()

        # Convolutional layers + ReLU + Dropout + MaxPool
        layers = []
        num_kernels = [1] + num_kernels
        for i in range(len(num_kernels) - 1):
            layers.append(
                nn.Conv1d(
                    in_channels=num_kernels[i],
                    out_channels=num_kernels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    padding=padding,
                )
            )
            layers.extend([nn.BatchNorm1d(num_kernels[i + 1]), nn.ReLU()])

        self.block = nn.Sequential(*layers)

        conv_out_size = int(
            calc_conv1d_output_size(
                input_dim, kernel_sizes, stride=stride, padding=padding
            )
            * num_kernels[-1]
        )

        self.out_layer = nn.Sequential(
            nn.Flatten(), nn.Linear(conv_out_size, output_dim)
        )  # Projection Layer

    def forward(self, x):
        # `x` must have input shape matching the requirement of `nn.Conv1d`: (N, C_in, L_in)
        cnn_out = self.block(x)

        return self.out_layer(cnn_out)
