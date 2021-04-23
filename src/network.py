# This file contains all of our DQN agents with 3 different perspectives

import torch
from torch import nn

class DQN(nn.Module):
  """
  DQN that takes on image as input and outputs the approximated q-values for each of the actions.
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_dim
           
        ## Check image input dimensions
        if h != 256:
            raise ValueError(f"Expecting input height: 256, got: {h}")
        if w != 256:
            raise ValueError(f"Expecting input width: 256, got: {w}")
    
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=3), # -> 32x64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2,padding=1), # -> 64x32x32
            nn.ReLU(),
            nn.Conv2d(4, 32, kernel_size=3, stride=2,padding=1), # -> 32x16x16
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, input):
        return self.net(input)

class DQNTwo(nn.Module):
    """
    DQN that takes two images as input and ouputs the approximated q-values for each of the actions.
    (conv2D + relu) x 3 for both images -> concatenate -> flatten -> (dense + relu) x 2 -> output 
    """

    def __init__(self, input_shape, n_actions):
        super(DQNTwo, self).__init__()
           
        ## Check image input dimensions
        if input_shape[1] != 256:
            raise ValueError(f"Expecting input height: 256, got: {input_shape[1]}")
        if input_shape[2] != 256:
            raise ValueError(f"Expecting input width: 256, got: {input_shape[2]}")

        self.convOne = nn.Sequential(
            ## CNN block for one image
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.convTwo = nn.Sequential(
            ## CNN block for second image (done so that they don't share weights)
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            ## Fully connected block
            nn.Flatten(),
            nn.Linear(16384, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, state):
        image1, image2 = state
        
        ## Output from each of the two images passed through the 3 CNNs
        out_one = self.convOne(image1).view(image1.size()[0], -1)
        out_two = self.convTwo(image2).view(image2.size()[0], -1)
        
        ## Concatenate the ouput then pass through the fully connected block
        combined = torch.cat((out_one.view(out_one.size(0), -1),
                              out_two.view(out_two.size(0), -1)), dim=1)


        return self.fc(combined)

