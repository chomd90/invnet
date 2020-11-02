from typing import Tuple

from didyprog.didyprog.reference.local import operators

import numpy as np


def sp_value(A : np.ndarray,operator: str ='hardmax') \
        -> float:
    """
    Djikstra operator.

    :param theta: _numpy.ndarray, shape = (T, S, S),
        Holds the potentials of the linear
         chain CRF
    :param operator: str in {'hardmax', 'softmax', 'sparsemax'},
        Smoothed max-operator
    :return: float,
        DTW value $Vit(\theta)$
    """
    return sp_grad(A,operator)[0]


def sp_grad(theta: np.ndarray,
                  adj_map: dict,
                  reverse_adj_map: dict,
                  operator: str='softmax')\
        ->Tuple[float,np.ndarray,np.ndarray,np.ndarray]:
    operator=operators[operator]
    hard_operator = operators['hardmax']
    n_nodes=theta.shape[0]
    assert n_nodes>=1
    V=np.zeros((n_nodes+1))
    V_hard = np.zeros((n_nodes + 1))
    Q=np.zeros((n_nodes,4))
    E_hat=np.zeros((n_nodes+1))
    E = np.zeros((n_nodes, 4))

    V[-1]=float('inf')
    V[-2]=0
    V_hard[-1]=float('inf')
    V_hard[-2]=0
    for i in reversed(range(n_nodes-1)):

        idxs=adj_map[i]
        fixed_idxs=[]
        for p in range(len(idxs)):
            if idxs[p] is None:
                fixed_idxs.append(n_nodes)
            else:
                fixed_idxs.append(idxs[p])
        options=V[fixed_idxs]+theta[i,:]
        V[i],Q[i]=operator.min(options)
        if np.isnan(V[i]):
            V[i]=float('inf')
            Q[i]=np.array([0]*4)
        hard_options = V_hard[fixed_idxs] + theta[i, :]
        V_hard[i], _ = hard_operator.min(hard_options)
        if np.isnan(V_hard[i]):
            V_hard[i] = float('inf')
    v=V[0]
    v_hard=V_hard[0]
    E[0]=Q[0]
    E_hat[0]=1

    for i in range(1,n_nodes):
        e_i,se_i,s_i,sw_i=reverse_adj_map[i]   #(RIP left_i)
        e_child=Q[e_i][0]*E_hat[e_i] if e_i is not None else 0
        se_child=Q[se_i][1]*E_hat[se_i] if se_i is not None else 0
        s_child=Q[s_i][2]*E_hat[s_i] if s_i is not None else 0
        sw_child=Q[sw_i][3]*E_hat[sw_i] if sw_i is not None else 0
        if e_i is not None:
            E[e_i][0]=e_child
        if se_i is not None:
            E[se_i][1]=se_child
        if s_i is not None:
            E[s_i][2]=s_child
        if sw_i is not None:
            E[sw_i][3] = sw_child
        E_hat[i]= e_child+se_child+s_child+sw_child
    return v,E,Q,E_hat,v_hard

if __name__=='__main__':
    pass