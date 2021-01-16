from didyprog.image_generation.graph_layer import GraphLayer
import torch

def test_graph_layer():

    ranged = torch.arange(2).view(1,1,-1)
    image = ranged.repeat((2, 2, 1))
    solution=torch.tensor([[1,           1,              0,-1*float('inf')                    ],
                          [-1*float('inf'),-1*float('inf'),4               , 1               ],
                           [1,-1*float('inf'),-1*float('inf'),-1*float('inf')],
                           [-1*float('inf'),-1*float('inf'),-1*float('inf'),-1*float('inf')]],
                          dtype=torch.float)
    solution=solution.view((1,4,4)).repeat((2,1,1))
    graph_layer = GraphLayer.apply
    theta = graph_layer(image)
    for batch in range(2):
        for i in range(4):
            for j in range(4):
                assert theta[batch,i,j]==solution[batch,i,j]