import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2),  # 12 * 12 * 8
            nn.ReLU(),

            nn.Conv2d(8, 16, 3, stride=2),  # 5 * 5 * 16
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2),  # 2 * 2 * 32
            nn.ReLU()
        )
    def forward(self, x):
        return self.layers(x)