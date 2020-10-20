from typing import Tuple

from didyprog.reference.local import operators

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
    n_nodes=theta.shape[0]
    assert n_nodes>=1
    V=np.zeros((n_nodes+1))
    Q=np.zeros((n_nodes,2))
    E_hat=np.zeros((n_nodes+1))
    E = np.zeros((n_nodes, 2))

    V[-1]=float('inf')
    V[-2]=0
    for i in reversed(range(n_nodes-1)):
        idxs=adj_map[i]
        for p in range(len(idxs)):
            if idxs[p] is None:
                idxs[p]=n_nodes
        options=V[idxs]+theta[i,:]
        V[i],Q[i]=operator.min(options)
        if np.isnan(V[i]):
            V[i]=float('inf')
            Q[i]=np.array([0]*2)
    v=V[0]
    E[0]=Q[0]
    E_hat[0]=1
    for i in range(1,n_nodes):
        up_i,left_i=reverse_adj_map[i]   #(RIP left_i)
        up_child=Q[up_i][0]*E_hat[up_i] if up_i is not None else 0
        left_child=Q[left_i][1]*E_hat[left_i] if left_i is not None else 0
        if up_i is not None:
            E[up_i][0]=up_child
        if left_i is not None:
            E[left_i][1]=left_child
        E_hat[i]= up_child+left_child
    return v,E,Q,E_hat

if __name__=='__main__':
    pass