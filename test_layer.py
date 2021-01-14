from didyprog.image_generation.sp_layer import SPLayer,hard_v
import torchvision
from torchvision import transforms, datasets
from models.wgan import *
import torch
from didyprog.image_generation.mnist_digit import make_graph

input=torch.randn(size=[32,32])
input.requires_grad=True
dp_layer=SPLayer.apply

output=dp_layer(input)

output.backward()

print('hello')