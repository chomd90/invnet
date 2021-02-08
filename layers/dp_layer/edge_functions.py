import torch

def sum_squared(V_1,V_2):
    '''
    V_1: [batch_size,max_i,max_j,1]
    V_2: [batch_size,max_i,max_j,4]
    '''
    return (V_1+V_2)**2

def diff_squared(V_1,V_2):
    '''
    V_1: [batch_size,max_i,max_j,1]
    V_2: [batch_size,max_i,max_j,4]
    '''
    return (V_1-V_2)**2

def diff_exp(V_1,V_2):
    '''
    V_1: [batch_size,max_i,max_j,1]
    V_2: [batch_size,max_i,max_j,4]
    '''
    return torch.exp(V_1-V_2)

def v1_only(V_1,V_2):
    return torch.cat(4*[V_1],dim=3)

edge_f_dict={'sum_squared':sum_squared,'diff_squared':diff_squared,'diff_exp':diff_squared,'v1_only':v1_only}