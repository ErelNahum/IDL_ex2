import torch.nn as nn


class InverseFCLayer(nn.Module):
    def __init__(self, latent_dimension: int = 12) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dimension, 128),
            nn.Unflatten(1, (32, 2, 2))
        )

    def forward(self, x):
        return self.layers(x)
