import numpy as np
from didyprog.reference.shortest_path import sp_grad
from image_generation.sp_utils import compute_diff
from image_generation.mnist_digit import make_graph

class SPLayer:

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
        theta, loc_to_idx, idx_to_loc, adj_map, rev_map = make_graph(image)
        v, E, Q, E_hat = sp_grad(theta, adj_map, rev_map)
        return v,E,idx_to_loc

    def backward(self,image, E, idx_to_loc):
        '''
        Parameters
        ----------
        image: numpy.ndarray
         shape nxn
        E: numpy.ndarray
         Shape of nxnx2 - Gradient of the loss with respect to the edge weights
        idx_to_loc: list
         Gives the i,j location of each node based on index
        Returns
        -------
        full_grad: numpy.ndarray
         Gradient of the loss with respect to the pixel intensities
        '''
        below, to_right = compute_diff(image)
        # below_{i,j} = P_{i,j} - P_{i+1,j}

        max_i, max_j = image.shape

        local_grad_forward = np.zeros((max_i, max_j, 2))
        for idx, location in enumerate(idx_to_loc):
            i, j = location
            local_grad_forward[i, j] = E[idx]

        below_derivative = 2 * below
        right_derivative = 2 * to_right

        forward_effect = np.stack([below_derivative, right_derivative], axis=2)

        forward_grad = local_grad_forward * forward_effect
        back_grad = np.zeros_like(forward_grad)
        for idx, location in enumerate(idx_to_loc):
            i, j = location
            if i == 0:
                back_grad[i, j, 0] = 0
            else:
                back_grad[i, j, 0] = -1 * forward_grad[i - 1, j, 0]
            if j == 0:
                back_grad[i, j, 1] = 0
            else:
                back_grad[i, j, 1] = -1 * forward_grad[i, j - 1, 1]
        full_grad = (back_grad + forward_grad).sum(axis=2)
        return full_grad

    def true_value(self,image):
        theta, loc_to_idx, idx_to_loc, adj_map, rev_map = make_graph(image)
        v, E, Q, E_hat = sp_grad(theta, adj_map, rev_map,operator='hardmax')
        return v