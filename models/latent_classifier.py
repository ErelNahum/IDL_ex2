import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(12, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
    def forward(self, x):
        return self.layers(x)