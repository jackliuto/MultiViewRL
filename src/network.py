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

#########################
## Victoria's Networks ##
#########################

class DQNOne(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DQNOne, self).__init__()

        if input_shape[1] != 256:
            raise ValueError(f"Expecting input height: 256, got: {input_shape[1]}")
        if input_shape[2] != 256:
            raise ValueError(f"Expecting input width: 256, got: {input_shape[2]}")

        self.conv = nn.Sequential(
            ## haven't played around with any parameters here - used old ones
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, image):
        return self.conv(image)


class DQNTwo(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DQNTwo, self).__init__()

        if input_shape[1] != 256:
            raise ValueError(f"Expecting input height: 256, got: {input_shape[1]}")
        if input_shape[2] != 256:
            raise ValueError(f"Expecting input width: 256, got: {input_shape[2]}")

        self.convOne = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.convTwo = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16384, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, state):
        image1, image2 = state
        out_one = self.convOne(image1).view(image1.size()[0], -1)
        out_two = self.convTwo(image2).view(image2.size()[0], -1)

        combined = torch.cat((out_one.view(out_one.size(0), -1),
                              out_two.view(out_two.size(0), -1)), dim=1)

        ## could pass through another convolution layer after combining them

        return self.fc(combined)

