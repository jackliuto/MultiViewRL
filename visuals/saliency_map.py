import torch
from torch import nn
import os
import cv2
from captum.attr import GuidedGradCam, Saliency
import matplotlib.pyplot as plt
import numpy as np

#Model for DQN
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

    def load(self, load_path):
        if os.path.isfile(load_path):
            checkpoint = torch.load(load_path)
            
            checkpoint['model']['0.weight']=checkpoint['model']['net.0.weight']
            checkpoint['model']['0.bias']=checkpoint['model']['net.0.bias']

            checkpoint['model']['2.weight']=checkpoint['model']['net.2.weight']
            checkpoint['model']['2.bias']=checkpoint['model']['net.2.bias']

            checkpoint['model']['4.weight']=checkpoint['model']['net.4.weight']
            checkpoint['model']['4.bias']=checkpoint['model']['net.4.bias']

            checkpoint['model']['7.weight']=checkpoint['model']['net.7.weight']
            checkpoint['model']['7.bias']=checkpoint['model']['net.7.bias']
            
            checkpoint['model']['9.weight']=checkpoint['model']['net.9.weight']
            checkpoint['model']['9.bias']=checkpoint['model']['net.9.bias']
            
            del checkpoint['model']['net.0.weight']
            del checkpoint['model']['net.0.bias']

            del checkpoint['model']['net.2.weight']
            del checkpoint['model']['net.2.bias']

            del checkpoint['model']['net.4.weight']
            del checkpoint['model']['net.4.bias']
            
            del checkpoint['model']['net.7.weight']
            del checkpoint['model']['net.7.bias']
            
            del checkpoint['model']['net.9.weight']
            del checkpoint['model']['net.9.bias']
            
            self.net.load_state_dict(checkpoint['model'])
            print("=> loaded checkpoint '{} ".format(load_path))
        else:
            print("=> no checkpoint found at '{}'".format(load_path))

dqn_net = DQN((3, 256, 256), 4)

dqn_net.load("hard_ego.chkpt")

img_bgr = cv2.imread("pictures/hard/ego/ego_0.png")

img = img_bgr[...,::-1].copy()

img_tensor = torch.tensor(img).float()

reshaped_tensor = torch.reshape(img_tensor, (1,3,256,256))

#Get predicted action
predicted = dqn_net(reshaped_tensor)
np_pred = predicted.detach().numpy()
cls = np.argmax(np_pred)

#Create saliency model from a CNN model
guided_gc = Saliency(dqn_net)

#Send input image into saliency model
#Attribution is the output map
attribution = guided_gc.attribute(reshaped_tensor, int(cls))

#Convert output map into RGB image
unreshaped_tensor = torch.reshape(attribution, (3,256,256))
tensor_img = unreshaped_tensor.view(unreshaped_tensor.shape[1], unreshaped_tensor.shape[2], unreshaped_tensor.shape[0])
tensor_img = tensor_img.detach().numpy()
tensor_img = ((tensor_img - tensor_img.min()) * (1/(tensor_img.max() - tensor_img.min()) * 255)).astype('uint8')

#View map
plt.figure(1)
plt.imshow(img)
plt.figure(2)
plt.imshow(tensor_img)
plt.show()

