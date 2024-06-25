import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2),  # 5 * 5 * 16

            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, stride=2),  # 13 * 13 * 8

            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2)  # OUTPUT: 28 * 28 * 1
        )
    def forward(self, x):
        return self.layers(x)