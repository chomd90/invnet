import pytest
import torch
import numpy as np
from layers.sp_layer import SPLayer
from layers.graph_layer import GraphLayer
from layers.sp_utils import adjacency
from scipy.optimize import check_grad
from layers.mnist_digit import make_graph
from didyprog.didyprog.reference.shortest_path import sp_forward,sp_grad

_,_,adj_map,rev_map=make_graph(2,2)
def make_data():
    a_range=torch.arange(1,3,dtype=torch.float,requires_grad=True).view((1,-1))
    a_range=torch.cat([a_range,a_range+2],dim=0).unsqueeze(0)
    a_range=a_range ** 2

    graph_layer = GraphLayer.apply
    thetas=graph_layer(a_range)
    return thetas

def apply_layers(X):
    sp_layer=SPLayer.apply
    X.retain_grad()
    lengths=sp_layer(X,adj_map,rev_map,dtype=torch.float64)
    output=lengths.sum()
    output.backward(retain_graph=True)

    grad=X.grad
    return output,grad.squeeze(0)


def test_sp_backward():
    thetas = make_data()

    def grad(X):
        X = torch.tensor(X).detach()
        X.requires_grad = True
        X = X.reshape(thetas.shape)

        output, grad = apply_layers(X)
        return grad.view(-1).detach().numpy()

    def func(X):
        Y = torch.tensor(X,dtype=torch.float64).detach()
        Y = Y.reshape(thetas.shape)
        output = sp_layer(Y, adj_map, rev_map).sum()
        np.array(output,dtype=np.float64)
        return output.numpy()

    err = check_grad(func, grad, (thetas.detach().numpy()).ravel())
    assert err < 1e-6

def test_sp_forward():
    thetas=make_data()
    v_hard,grad=apply_layers(thetas)
    true_output=819
    err=true_output-v_hard.item()
    assert err < 1e-6
sp_layer=SPLayer.apply

if __name__ == '__main__':
    unittest.main()
