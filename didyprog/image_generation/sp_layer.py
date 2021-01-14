import numpy as np
from didyprog.didyprog.reference.shortest_path import sp_grad,hard_sp
from didyprog.image_generation.sp_utils import compute_diff
from didyprog.image_generation.mnist_digit import make_graph,compute_distances
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
        pos_image= torch.sigmoid( input)
        pos_image.requires_grad=True

        image = input.clone().detach().cpu().numpy()
        theta = compute_distances(image, idx2loc, adj_map)
        v, E, Q, E_hat, v_hard = sp_grad(theta, adj_map, rev_map, 'softmax')
        v, E, v_hard,image = torch.tensor(v), torch.tensor(E), torch.tensor(v_hard), torch.tensor(image)
        ctx.save_for_backward(E, input, v_hard)
        v_hard.requires_grad=True
        return v_hard
        # theta= compute_distances(pos_image,self.idx2loc,self.adj_map)
        # v, E, Q, E_hat,v_hard = sp_grad(theta, self.adj_map, self.rev_map,'softmax')
        # ctx.save_for_backward(E)
        # ctx.save_for_backward(image)
        # return v,E,v_hard

    @staticmethod
    def backward(ctx,v_grad):
        #TODO Split this into graph creation derivative and shortest paths derivative
        '''v_grad is the gradient of the loss with respect to v_hard'''
        E, image, v_hard = ctx.saved_tensors
        minus_east, minus_se, minus_s, minus_sw = compute_diff(image, add=True)
        # below_{i,j} = P_{i,j} - P_{i+1,j}

        max_i, max_j = image.shape

        local_grad_forward = np.zeros((max_i, max_j, 4))
        """Local grad forward is E but indexed based on on location instead of index"""
        for idx, location in enumerate(idx2loc):
            i, j = location
            local_grad_forward[i, j] = E[idx]

        e_deriv = 2 * minus_east
        se_deriv = 2 * minus_se
        s_deriv = 2 * minus_s
        sw_deriv = 2 * minus_sw

        forward_effect = np.stack([e_deriv, se_deriv, s_deriv, sw_deriv], axis=2)

        forward_grad = local_grad_forward * forward_effect
        back_grad = np.zeros_like(forward_grad)
        for idx, location in enumerate(idx2loc):
            i, j = location

            if j > 0:  # Gradient from westward parent
                back_grad[i, j, 0] = forward_grad[i, j - 1, 0]

            if j > 0 and i > 0:  # Gradient from northwestern parent
                back_grad[i, j, 1] = forward_grad[i - 1, j - 1, 0]

            if i > 0:  # Gradient from northern parent
                back_grad[i, j, 2] = forward_grad[i - 1, j, 2]

            if i > 0 and j < max_j - 1:  # Gradient from northeast parent
                back_grad[i, j, 3] = forward_grad[i - 1, j + 1, 3]

        full_grad = (back_grad + forward_grad).sum(axis=2)
        output= torch.tensor(full_grad,dtype=torch.float)*v_grad
        return output.to(image.device)



def hard_v(image,idx2loc,adj_map):
    theta = compute_distances(image,idx2loc,adj_map)
    return hard_sp(theta,adj_map)