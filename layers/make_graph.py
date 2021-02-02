import random
import numpy as np
from collections import defaultdict

random.seed(0)
np.random.seed(0)


def make_graph(max_i,max_j):
    #TODO Rewrite this function nicely
    idx_to_loc=[]
    loc_to_idx={}
    total_nodes=0
    for i in range(max_i):
        for j in range(max_j):
            idx_to_loc.append((i,j))
            loc_to_idx[(i,j)]=total_nodes
            total_nodes+=1
    map = {}
    rev_map = defaultdict(lambda: [None, None, None, None])
    for i in range(total_nodes):
        i_loc,j_loc=idx_to_loc[i]

        on_bottom= i_loc==max_i-1
        on_right= j_loc==max_j-1
        on_left= (j_loc==0)
        if not on_right:
            east_idx = (i_loc,j_loc+1)
            east_i = loc_to_idx[east_idx]
        else:
            east_i=None

        if not (on_bottom or on_right):
            se_idx = (i_loc+1,j_loc+1)
            se_i = loc_to_idx[se_idx]
        else:
            se_i=None

        if not on_bottom:
            south_idx = (i_loc + 1, j_loc)
            south_i = loc_to_idx[south_idx]
        else:
            south_i=None

        if not (on_bottom or on_left):
            sw_idx=(i_loc+1,j_loc-1)
            sw_i = loc_to_idx[sw_idx]
        else:
            sw_i = None
        map[i] = [east_i, se_i, south_i,sw_i]
        rev_map[east_i][0] = i
        rev_map[se_i][1] = i
        rev_map[south_i][2] = i
        rev_map[sw_i][3] = i
    return loc_to_idx,idx_to_loc,map,rev_map



