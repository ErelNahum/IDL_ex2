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

class FCLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 12)
        )
    
    def forward(self, x):
        return self.layers(x)

class InverseFCLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(12, 128),
            nn.Unflatten(1, (32, 2, 2))
        )

    def forward(self, x):
        return self.layers(x)


class Classifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(12, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.Softmax()
        )
    def forward(self, x):
        return self.layers(x)
# class Classifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten(start_dim=0, end_dim=1)
#         self.linear_relu_stack = nn.Sequential(
#             Encoder(),
#             nn.Linear(12, 11),
#             nn.ReLU(),
#             nn.Linear(11, 10),
#             nn.Softmax()
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
