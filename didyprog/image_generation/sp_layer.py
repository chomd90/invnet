import numpy as np
from didyprog.didyprog.reference.shortest_path import sp_forward,sp_grad,hard_sp
from didyprog.image_generation.sp_utils import compute_distances,adjacency_function
from didyprog.image_generation.mnist_digit import make_graph
import torch
from torch.autograd import Function
import itertools

def sm(options):
    max_x=torch.max(options,dim=1)[0].view(-1,1)
    exp_x=torch.exp(options-max_x)
    Z=torch.sum(exp_x,dim=1).unsqueeze(-1)
    smooth_max=(torch.log(Z) + max_x).squeeze()
    probs=exp_x/Z
    smooth_max[smooth_max!=smooth_max]=-1*float('inf')
    probs[probs!=probs]=0
    return smooth_max,probs


class SPLayer(Function):

    def __init__(self):
        super(SPLayer,self).__init__()

    @staticmethod
    def forward(ctx,input,adj):
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
        thetas = input
        batch_size,n_nodes,_= thetas.shape
        assert n_nodes>1
        V=torch.zeros((batch_size,n_nodes+1)).to(input.device)
        #V is inverted here so that indexing can be used without inverting slices individually
        V_hard=torch.zeros((batch_size,n_nodes+1)).to(input.device)
        Q=torch.zeros((batch_size,n_nodes,4)).to(input.device)
        rev=torch.full((n_nodes,4),n_nodes,dtype=torch.long).to(input.device)
        for a in [V,V_hard]:
            a[:,-1],a[:-2]= -1*float('inf'),0
        for i in reversed(range(n_nodes-1)):
            theta=thetas[:,i,:]
            idxs=tuple(adj(i,replace=n_nodes))
            for dir,idx in enumerate(idxs):
                if idx<n_nodes:
                    rev[idx,dir]=i
            values=torch.stack([V[:,i] for i in idxs],dim=1)
            options=values+theta
            soft=sm(options)
            V[:,i],Q[:,i,:]=soft[0],soft[1]
            V_hard[:,i]=torch.max(options,dim=1)[0]
        v_hard=V_hard[:,0]
        ctx.save_for_backward(v_hard,Q,rev)
        return v_hard

    @staticmethod
    def backward(ctx,v_grad):
        #TODO Split this into graph creation derivative and shortest paths derivative
        '''v_grad is the gradient of the loss with respect to v_hard'''
        v_hard,Q,rev = ctx.saved_tensors
        b,n,_=Q.shape
        E_hat=torch.zeros((b,n+1),dtype=torch.float,device=Q.device)
        E = torch.zeros((b,n,4),dtype=torch.float,device=Q.device)

        E[:,0,:]=Q[:,0,:]
        E_hat[:,0]=1
        for i in range(1,n):
            back_idxs=rev[i].tolist()
            total=torch.zeros((b),dtype=torch.float,device=Q.device)
            for dir_idx,dir in enumerate(back_idxs):
                if dir <n:
                    parent=Q[:,i,dir_idx]*E_hat[:,dir_idx]
                    #E_hat is total effect of parent node on loss
                    #so parent represents the current node's effect on parent
                    total+=parent
                    E[:,i,dir_idx]=parent
            E_hat[:,i]=total
        return E,None

def hard_v(image,idx2loc,adj_map):
    theta = compute_distances(image)
    return hard_sp(np.array(theta),adj_map)