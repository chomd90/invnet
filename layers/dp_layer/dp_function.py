import torch
from torch.autograd import Function
class DPFunction(Function):

    def __init__(self):
        super(DPFunction, self).__init__()

    @staticmethod
    def forward(ctx, input, adj_array, rev_adj, max_op,replace):
        '''
            Parameters
            ----------
            input: numpy.ndarray
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
        if not ctx.needs_input_grad[0]:
            return DPFunction.hard_forward(input,adj_array,max_op,replace)
        op=DPFunction.s_min
        hard_op=torch.min
        if max_op:
            op=DPFunction.s_max
            hard_op=torch.max
        ctx.rev_map=rev_adj
        thetas = input
        batch_size,n_nodes,_= thetas.shape
        assert n_nodes>1
        V_hard=torch.zeros((batch_size,n_nodes+1)).to(input.device)

        V=torch.zeros((batch_size,n_nodes+1)).to(input.device)
        Q=torch.zeros((batch_size,n_nodes,4)).to(input.device)

        if replace==0:
            V[:,-1]=replace
            V_hard[:,-1]=replace
        else:
            V[:, -1] += replace
            V_hard[:, -1] += replace
        V[:-2] =  0
        V_hard[:-2]= 0
        for i in reversed(range(n_nodes-1)):
            theta=thetas[:,i,:]
            idxs=adj_array[i]
            for dir,idx in enumerate(idxs):
                if idx is None:
                    idxs[dir]=n_nodes
            values=torch.stack([V[:,i] for i in idxs],dim=1)
            options=values+theta
            soft=op(options)
            V[:,i],Q[:,i,:]=soft[0],soft[1]
            hard_values = torch.stack([V_hard[:, i] for i in idxs], dim=1)
            hard_options=hard_values+theta
            V_hard[:,i]=hard_op(hard_options,dim=1)[0]
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
        return E,None,None,None,None

    @staticmethod
    def s_max(options):
        max_x = torch.max(options, dim=1)[0].view(-1, 1)
        exp_x = torch.exp(options - max_x)
        Z = torch.sum(exp_x, dim=1).unsqueeze(-1)
        smooth_max = (torch.log(Z) + max_x).squeeze()
        probs = exp_x / Z
        return smooth_max, probs

    @staticmethod
    def s_min(options):
        neg_options = -1 * options
        s_min_val, s_argmin = DPFunction.s_max(neg_options)
        s_min_val *= -1
        return s_min_val, s_argmin

    @staticmethod
    def hard_forward(input, adj_array, max_op,replace):
        '''Computes v_hard as in forward(), but without any of the additional
        computation needed to make function differentiable'''
        hard_op=torch.min
        if max_op:
            hard_op=torch.max
        thetas = input
        batch_size,n_nodes,_= thetas.shape
        assert n_nodes>1
        V_hard=torch.zeros((batch_size,n_nodes+1)).to(input.device)

        if replace==0:
            V_hard[:,-1]=replace
        else:
            V_hard[:, -1] += replace
        V_hard[:-2]= 0
        for i in reversed(range(n_nodes-1)):
            theta=thetas[:,i,:]
            idxs=adj_array[i]
            for dir,idx in enumerate(idxs):
                if idx is None:
                    idxs[dir]=n_nodes
            hard_values = torch.stack([V_hard[:, i] for i in idxs], dim=1)
            hard_options=hard_values+theta
            V_hard[:,i]=hard_op(hard_options,dim=1)[0]
        v_hard=V_hard[:,0]

        return v_hard