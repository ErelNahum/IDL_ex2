import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(  # INPUT: 28 * 28 * 1
            nn.Conv2d(1, 8, 5, stride=2),  # 12 * 12 * 8
            nn.ReLU(),

            nn.Conv2d(8, 16, 3, stride=2),  # 5 * 5 * 16
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2),  # 2 * 2 * 32
            nn.ReLU(),

            nn.Flatten(0),
            nn.Linear(128, 12)  # OUTPUT: 12
        )

        self.decoder = nn.Sequential(  # INPUT: 12
            nn.Linear(12, 128),
            nn.Unflatten(0, [2, 2, 32]),  # 2 * 2 * 32

            nn.ConvTranspose2d(32, 16, 3, stride=2),  # 5 * 5 * 16
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, 3, stride=2),  # 12 * 12 * 8
            nn.ReLU(),

            nn.ConvTranspose2d(8, 1, 5, stride=2),  # OUTPUT: 28 * 28 * 1
            nn.Sigmoid()
        )

        def forward(self, x):
            encoded = self.encoder(x)
            return self.decoder(encoded)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten(start_dim=0, end_dim=1)
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(12, 180),
                nn.Con,
                nn.Linear(180, 180),
                nn.ReLU(),
                nn.Linear(180, 784),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten(start_dim=0, end_dim=1)
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(180, 180),
                nn.ReLU(),
                nn.Linear(180, 180),
                nn.ReLU(),
                nn.Linear(180, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten(start_dim=0, end_dim=1)
            self.linear_relu_stack = nn.Sequential(
                Encoder(),
                nn.Linear(12, 11),
                nn.ReLU(),
                nn.Linear(11, 10),
                nn.Softmax()
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits
