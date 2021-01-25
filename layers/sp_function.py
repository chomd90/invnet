import torch
from torch.autograd import Function

def s_max(options):
    max_x=torch.max(options,dim=1)[0].view(-1,1)
    exp_x=torch.exp(options-max_x)
    Z=torch.sum(exp_x,dim=1).unsqueeze(-1)
    smooth_max=(torch.log(Z) + max_x).squeeze()
    probs=exp_x/Z
    return smooth_max,probs

def s_min(options):
    neg_options=-1*options
    s_min_val,s_argmin= s_max(neg_options)
    s_min_val*=-1
    return s_min_val,s_argmin


class SPFunction(Function):

    def __init__(self):
        super(SPFunction,self).__init__()

    @staticmethod
    def forward(ctx,input,adj_map,rev_map,max_op):
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
        op=s_min
        hard_op=torch.min
        if max_op:
            op=s_max
            hard_op=torch.max
        ctx.rev_map=rev_map
        thetas = input
        batch_size,n_nodes,_= thetas.shape
        assert n_nodes>1
        V=torch.zeros((batch_size,n_nodes+1)).to(input.device)
        V_hard=torch.zeros((batch_size,n_nodes+1)).to(input.device)
        Q=torch.zeros((batch_size,n_nodes,4)).to(input.device)
        for a in [V,V_hard]:
            a[:,-1],a[:-2]= -1*float('inf'),0
        for i in reversed(range(n_nodes-1)):
            theta=thetas[:,i,:]
            idxs=adj_map[i]
            for dir,idx in enumerate(idxs):
                if idx is None:
                    idxs[dir]=n_nodes
            values=torch.stack([V[:,i] for i in idxs],dim=1)
            options=values+theta
            soft=op(options)
            V[:,i],Q[:,i,:]=soft[0],soft[1]
            V_hard[:,i]=hard_op(options,dim=1)[0]
        v_hard=V_hard[:,0]
        ctx.save_for_backward(v_hard,Q)
        return v_hard

    @staticmethod
    def backward(ctx,v_grad):
        '''v_grad is the gradient of the loss with respect to v_hard'''
        v_hard,Q = ctx.saved_tensors
        b,n,_=Q.shape
        E_hat=torch.zeros((b,n),dtype=torch.float,device=Q.device)
        E = torch.zeros((b,n,4),dtype=torch.float,device=Q.device)

        E[:,0,:]=Q[:,0,:]
        E_hat[:,0]=1
        for i in range(1,n):
            back_idxs=ctx.rev_map[i]
            total=torch.zeros((b),dtype=torch.float,device=Q.device)
            for dir_idx,back_idx in enumerate(back_idxs):
                if back_idx is not None and dir_idx <n:
                    parent=Q[:,back_idx,dir_idx]*E_hat[:,back_idx]
                    #E_hat is total effect of parent node on loss
                    #so parent represents the current node's effect on parent
                    total+=parent
                    E[:,back_idx,dir_idx]=parent
            E_hat[:,i]=total
        return E,None,None,None