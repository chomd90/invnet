from didyprog.image_generation.sp_layer import SPLayer,hard_v
from didyprog.image_generation.graph_layer import GraphLayer
import torchvision
from torchvision import transforms, datasets
from models.wgan import *
import torch
from didyprog.image_generation.mnist_digit import make_graph

input=torch.randn(size=[32,32])
dp_layer=SPLayer.apply
graph_layer=GraphLayer.apply

theta=graph_layer(input)
output=dp_layer(theta)

output.backward()

print('hello')