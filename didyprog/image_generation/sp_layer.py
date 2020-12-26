import numpy as np
from didyprog.didyprog.reference.shortest_path import sp_grad,hard_sp
from didyprog.image_generation.sp_utils import compute_diff
from didyprog.image_generation.mnist_digit import make_graph,compute_distances
import math
import torch
import torch.nn as nn


#TODO figure out ctx.store_for_backward()

class SPLayer(nn.Module):

    def __init__(self):
        super(SPLayer,self).__init__()
        _,self.idx2loc,self.adj_map,self.rev_map=make_graph(32,32)

    def forward(self,image):
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

        pos_image= torch.tensor( 1/(1+np.exp(-image)) )
        pos_image.requires_grad=True
        autograd_function = self.create_autograd_function().apply

        return autograd_function(pos_image)

        theta= compute_distances(pos_image,self.idx2loc,self.adj_map)
        v, E, Q, E_hat,v_hard = sp_grad(theta, self.adj_map, self.rev_map,'softmax')
        ctx.save_for_backward(E)
        ctx.save_for_backward(image)
        return v,E,v_hard

    def create_autograd_function(self):

        class dp_function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, image):
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
                image = image.detach().numpy()
                theta = compute_distances(image, self.idx2loc, self.adj_map)
                v, E, Q, E_hat, v_hard = sp_grad(theta, self.adj_map, self.rev_map, 'softmax')
                ctx.save_for_backward(torch.tensor(E),torch.tensor(image))
                v,E,v_hard = torch.tensor(v),torch.tensor(E),torch.tensor(v_hard)
                return v, E, v_hard

            @staticmethod
            def backward(ctx, v,E,v_hard):
                '''
                Parameters
                ----------
                image: numpy.ndarray
                 shape nxn
                E: numpy.ndarray
                 Shape of nx4 - Gradient of the loss with respect to the edge weights
                idx_to_loc: list
                 Gives the i,j location of each node based on index
                Returns
                -------
                full_grad: numpy.ndarray
                 Gradient of the loss with respect to the pixel intensities
                '''

                image,E =ctx.saved_tensors
                minus_east, minus_se, minus_s, minus_sw = compute_diff(image, add=True)
                # below_{i,j} = P_{i,j} - P_{i+1,j}

                max_i, max_j = image.shape

                local_grad_forward = np.zeros((max_i, max_j, 4))
                """Local grad forward is E but indexed based on on location instead of index"""
                for idx, location in enumerate(self.idx2loc):
                    i, j = location
                    local_grad_forward[i, j] = E[idx]

                e_deriv = 2 * minus_east
                se_deriv = 2 * minus_se
                s_deriv = 2 * minus_s
                sw_deriv = 2 * minus_sw

                forward_effect = np.stack([e_deriv, se_deriv, s_deriv, sw_deriv], axis=2)

                forward_grad = local_grad_forward * forward_effect
                back_grad = np.zeros_like(forward_grad)
                for idx, location in enumerate(self.idx2loc):
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
                return full_grad

        return dp_function

def hard_v(image,idx2loc,adj_map):
    theta = compute_distances(image,idx2loc,adj_map)
    return hard_sp(theta,adj_map)