from layers.sp_utils import compute_diff
import torch
from torch.autograd import Function
import torch.nn as nn

class GraphLayer(nn.Module):

    def __init__(self):
        super(GraphLayer,self).__init__()

    def forward(self,input):
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
        images = input
        edge_vals = torch.stack(compute_diff(images, replace=-1 * float('inf')), dim=3)
        #TODO Allow graph computation for arbitrary edge weight function
        edge_sums_flattened = torch.flatten(edge_vals, 1, 2)
        thetas = edge_sums_flattened ** 2
        thetas[thetas == float('inf')] = -1 * float('inf')
        return thetas


