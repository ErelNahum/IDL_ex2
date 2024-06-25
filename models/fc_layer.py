import torch.nn as nn

class FCLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 12)
        )
    
    def forward(self, x):
        return self.layers(x)