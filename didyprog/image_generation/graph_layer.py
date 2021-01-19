from didyprog.image_generation.sp_utils import compute_diff
import torch
from torch.autograd import Function
from didyprog.image_generation.sp_utils import compute_distances

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
        image = input
        ctx.save_for_backward(input)
        theta = compute_distances(image)
        return theta

    @staticmethod
    def backward(ctx,E):
        images = ctx.saved_tensors[0]
        batch_size,max_i, max_j = images.shape

        local_grad_forward = E.view(batch_size,max_i,max_j,4)

        minus_east, minus_se, minus_s, minus_sw = compute_diff(images, add=True)
        e_deriv = 2 * minus_east
        se_deriv = 2 * minus_se
        s_deriv = 2 * minus_s
        sw_deriv = 2 * minus_sw

        forward_effect = torch.stack([e_deriv, se_deriv, s_deriv, sw_deriv], axis=3)

        forward_grad = local_grad_forward * forward_effect

        west_grad=torch.roll(e_deriv,1,2)
        west_grad[:,:,0]=0

        nw_grad=torch.roll(se_deriv,1,1)
        nw_grad[:,0,:]=0
        nw_grad=torch.roll(nw_grad,1,2)
        nw_grad[:,:,0]=0

        north_grad=torch.roll(s_deriv,1,1)
        north_grad[:,0,:]=0

        ne_grad = torch.roll(sw_deriv, 1, 1)
        ne_grad[:, 0, :]=0
        ne_grad = torch.roll(ne_grad, -1, 2)
        ne_grad[:, :, -1]=0

        back_effect=torch.stack([west_grad,nw_grad,north_grad,ne_grad],axis=3)


        full_grad = (back_effect + forward_grad).sum(axis=3)
        output= full_grad
        return output

