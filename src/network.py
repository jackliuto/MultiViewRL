import torch
from torch import nn

class DQN(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 256:
            raise ValueError(f"Expecting input height: 256, got: {h}")
        if w != 256:
            raise ValueError(f"Expecting input width: 256, got: {w}")

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=3), # -> 32x64x64
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,padding=1), # -> 64x32x32
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,padding=1), # -> 32x16x16
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, input):
        return self.net(input)

