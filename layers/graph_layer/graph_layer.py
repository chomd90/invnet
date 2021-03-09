import torch
import torch.nn as nn


class GraphLayer(nn.Module):

    def __init__(self,null,edge_f,make_pos):
        self.null=null
        self.edge_f=edge_f
        self.make_positive=make_pos
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
        images=input
        if self.make_positive:
            images = torch.exp(input) #Make all the values positive

        b,max_i,max_j=images.shape
        shift_lst=[(0,1,),(1,1,),(1,0),(1,-1)]
        shifted_images=torch.stack([self.shifted(images,shifts) for shifts in shift_lst],dim=3)
        thetas=self.edge_f(images.unsqueeze(-1),shifted_images)#.view(b,max_i*max_j,4)
        thetas=self.replace_null(thetas)
        thetas=thetas.view(b,max_i*max_j,4)
        return thetas

    def shifted(self, images,shifts):
        shift_i,shift_j=shifts
        shifted=torch.roll(images,[-shift_i,-shift_j],[1,2])
        return shifted

    def replace_null(self,thetas):
        output=thetas.clone()
        output[:, :, -1, :2] = self.null
        output[:, -1, :, 1:4] = self.null
        output[:, :, 0, 3] = self.null
        return output
