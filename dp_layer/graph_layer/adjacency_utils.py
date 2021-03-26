def idxloc(size_j,idx):
    return idx//size_j,idx%size_j

def locidx(size_j,idx_i,idx_j):
    return size_j*idx_i + idx_j


def adjacency_function(max_i,max_j):
    def adjacency(idx, replace=None):
        i, j = idxloc(max_j, idx)
        if j < max_j - 1:
            yield idx + 1
            if i < max_i - 1:
                yield idx + max_j + 1
            else:
                yield replace
        else:
            yield replace
            yield replace
        if i < max_i - 1:
            yield idx + max_j
            if j > 0:
                yield idx + max_j - 1
            else:
                yield replace
        else:
            yield replace
            yield replace
    return adjacency


def idx_adjacency(max_i,max_j):
    adj_array=[[None for dir in range(4)] for node in range(max_i*max_j)]
    rev_array=[[None for dir in range(4)] for node in range(max_i*max_j)]
    adj_f=adjacency_function(max_i,max_j)
    for node_idx in range(max_i*max_j):
        nexts=list(adj_f(node_idx))
        adj_array[node_idx]=nexts
        for dir,next in enumerate(nexts):
            if next is not None:
                rev_array[next][dir]=node_idx
    return adj_array,rev_array
