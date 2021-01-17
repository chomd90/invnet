import numpy as np
from didyprog.didyprog.reference.shortest_path import sp_forward,sp_grad,hard_sp
from didyprog.image_generation.sp_utils import compute_diff,compute_distances
from didyprog.image_generation.mnist_digit import make_graph
import math
import torch
import torch.nn as nn
from torch.autograd import Function,Variable

_,idx2loc,adj_map,rev_map=make_graph(32,32)

class SPLayer(Function):

    def __init__(self):
        super(SPLayer,self).__init__()

    @staticmethod
    def forward(ctx,input):
        '''
            Parameters
            ----------
            image: numpy.ndarray
             shape nxn
            Returns
            -------
            v: int
             Shortest path value computed by soft-DP
            image_grad: numpy.ndarray
             Gradient of the loss w.r.t the image pixel values
            true_shortest_path: int
             Shortest path value computed by hard-DP
            '''
        theta = input.clone().detach().cpu().numpy()
        v_hard,v,Q = sp_forward(theta, adj_map, 'softmax')
        v_hard,v,Q=torch.tensor(v_hard).to(input.device),torch.tensor(v),torch.tensor(Q)
        ctx.save_for_backward(v_hard,v,Q)
        return v_hard

    @staticmethod
    def backward(ctx,v_grad):
        #TODO Split this into graph creation derivative and shortest paths derivative
        '''v_grad is the gradient of the loss with respect to v_hard'''
        v_hard,v,Q = ctx.saved_tensors
        E, E_hat =sp_grad(Q,rev_map)
        return torch.tensor(E,dtype=torch.float).to(v_hard.device)

def hard_v(image,idx2loc,adj_map):
    theta = compute_distances(image)
    return hard_sp(np.array(theta),adj_map)