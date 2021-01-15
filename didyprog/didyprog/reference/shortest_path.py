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


def sp_forward(theta: np.ndarray,
                  adj_map: dict,
                  operator: str='softmax')\
        ->Tuple[float,np.ndarray,np.ndarray,np.ndarray]:
    operator=operators[operator]
    hard_operator = operators['hardmax']
    n_nodes=theta.shape[0]
    assert n_nodes>=1
    V=np.zeros((n_nodes+1))
    V_hard = np.zeros((n_nodes + 1))
    Q=np.zeros((n_nodes,4))

    for a in [V,V_hard]:
        a[-1],a[-2]  = -1 * float('inf'),0
    for i in reversed(range(n_nodes-1)):
        idxs=adj_map[i]
        idxs= [idx if not idx is None else n_nodes for idx in idxs]
        options=V[idxs]+theta[i,:]
        V[i],Q[i]=operator.max(options)
        if np.isnan(V[i]):
            V[i]=-1*float('inf')
            Q[i]=np.array([0]*4)
        hard_options = V_hard[idxs] + theta[i, :]
        V_hard[i], _ = hard_operator.max(hard_options)
        if np.isnan(V_hard[i]):
            V_hard[i] = -1*float('inf')
    v=V[0]
    v_hard=V_hard[0]
    return v_hard,v,Q

def sp_grad(Q,rev_adj_map):
    n_nodes=Q.shape[0]
    E_hat=np.zeros((n_nodes+1))
    E = np.zeros((n_nodes, 4))

    E[0]=Q[0]
    E_hat[0]=1
    for i in range(1,n_nodes):
        directions=rev_adj_map[i]
        total=0
        for j,direction in enumerate(directions):
            child=Q[direction][j]*E_hat[direction] if direction is not None else 0
            total+=child
            if direction is not None:
                E[direction]=child
        E_hat[i]=total
    return E,E_hat

def hard_sp(theta: np.ndarray,
                  adj_map: dict,)\
        ->Tuple[float,np.ndarray]:
    hard_operator = operators['hardmax']
    n_nodes=theta.shape[0]
    assert n_nodes>=1
    V_hard = np.zeros((n_nodes + 1))

    V_hard[-1]=-1*float('inf')
    V_hard[-2]=0
    for i in reversed(range(n_nodes-1)):
        idxs=adj_map[i]
        fixed_idxs=[]
        for p in range(len(idxs)):
            if idxs[p] is None:
                fixed_idxs.append(n_nodes)
            else:
                fixed_idxs.append(idxs[p])
        hard_options = V_hard[fixed_idxs] + theta[i, :]
        V_hard[i], _ = hard_operator.max(hard_options)
        if np.isnan(V_hard[i]):
            V_hard[i] = -1*float('inf')
    v_hard=V_hard[0]
    return v_hard

if __name__=='__main__':
    pass