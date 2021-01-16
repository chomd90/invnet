from didyprog.image_generation.graph_layer import GraphLayer
import torch

def test_graph_layer():

    range = torch.arange(4).view(1,-1)
    image=range.repeat((4,1))
    graph_layer = GraphLayer.apply

    theta = graph_layer(image)

    assert True