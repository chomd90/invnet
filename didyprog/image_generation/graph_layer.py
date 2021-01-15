import numpy as np
from didyprog.didyprog.reference.shortest_path import sp_forward,sp_grad,hard_sp
from didyprog.image_generation.sp_utils import compute_diff
from didyprog.image_generation.mnist_digit import make_graph,compute_distances
import torch
from torch.autograd import Function,Variable

_,idx2loc,adj_map,rev_map=make_graph(32,32)

class GraphLayer(Function):

    def __init__(self):
        super(GraphLayer,self).__init__()

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
        image = input.clone().detach().cpu().numpy()
        ctx.save_for_backward(input)
        theta = compute_distances(image, idx2loc, adj_map)
        return torch.tensor(theta,dtype=torch.float)

    @staticmethod
    def backward(ctx,E):
        image = ctx.saved_tensors[0]
        max_i, max_j = image.shape

        local_grad_forward = np.zeros((max_i, max_j, 4))
        """Local grad forward is E but indexed based on on location instead of index"""
        for idx, location in enumerate(idx2loc):
            i, j = location
            local_grad_forward[i, j] = E[idx]

        minus_east, minus_se, minus_s, minus_sw = compute_diff(image, add=True)

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
        output= torch.tensor(full_grad,dtype=torch.float)
        return output.to(image.device)

