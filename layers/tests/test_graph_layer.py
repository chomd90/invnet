from layers.graph_layer import GraphLayer
import torch
from scipy.optimize import check_grad
import numpy as np

def make_data():
    ranged = torch.arange(2, dtype=torch.float64).view(1, 1, -1)
    image = ranged.repeat((2, 2, 1))
    image.requires_grad = True
    solution = torch.tensor([[1, 1, 0, -1 * float('inf')],
                             [-1 * float('inf'), -1 * float('inf'), 4, 1],
                             [1, -1 * float('inf'), -1 * float('inf'), -1 * float('inf')],
                             [-1 * float('inf'), -1 * float('inf'), -1 * float('inf'), -1 * float('inf')]],
                            dtype=torch.float64)
    solution = solution.view((1, 4, 4)).repeat((2, 1, 1))
    return image, solution

def test_graph_layer():

    image,solution=make_data()
    graph_layer = GraphLayer()
    theta = graph_layer(image)
    theta.sum().backward()

    for batch in range(2):
        for i in range(4):
            for j in range(4):
                assert theta[batch,i,j]==solution[batch,i,j]

def test_sp_backward():
    images ,_ = make_data()

    def grad(X):
        graph_layer=GraphLayer()
        X = torch.tensor(X).detach()
        X.requires_grad = True
        X = X.reshape(images.shape)
        output=graph_layer(X)[0,0,0]
        X.retain_grad()
        output.backward()
        grad=X.grad.view(-1)
        return grad.detach().numpy()

    def func(X):
        graph_layer = GraphLayer()
        Y = torch.tensor(X,dtype=torch.float64).detach()
        Y = Y.reshape(images.shape)
        output = graph_layer(Y)[0,0,0]
        output=np.array(output,dtype=np.float64)
        return output

    err = check_grad(func, grad, (images.detach().numpy()).ravel())
    assert err < 1e-6
