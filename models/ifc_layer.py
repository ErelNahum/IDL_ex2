import torch.nn as nn

class InverseFCLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(12, 128),
            nn.Unflatten(1, (32, 2, 2))
        )

    def forward(self, x):
        return self.layers(x)
