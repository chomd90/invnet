import numpy as np
import torch
def compute_distances(images):
    batch_size,max_i,max_j=images.shape
    edge_sums = torch.stack(compute_diff(images, replace=-1 * float('inf')), dim=3)
    edge_sums_flattened=torch.flatten(edge_sums,1,2)
    theta=edge_sums_flattened**2
    theta[theta==float('inf')]=-1*float('inf')
    return theta

def compute_diff(image,add=True,replace=0):
    '''
    Parameters
    ----------
    image: numpy.ndarray of shape nxn
    Returns
    -------
    below: numpy.ndarray represents image[i][j]-image[i+1][j]
    to_right: numpy.ndarray represents image[i][j]-image[i][j+1]
    '''
    coeff=-1 if add else 1
    device=image.device
    batch_size,max_i,max_j=image.shape

    zips= torch.zeros((batch_size,1,max_j),dtype=torch.float).to(device)
    zips_above=torch.cat([zips,image],dim=1)
    zips_below=torch.cat([image,zips], dim=1)
    minus_below = zips_above - coeff*zips_below
    minus_below = minus_below[:,1:, :]
    minus_below[:,-1, :] = replace

    zips = torch.zeros((batch_size,max_i,1),dtype=torch.float).to(device)
    right_side = torch.cat([image, zips], axis=2)
    left_side = torch.cat([zips, image], axis=2)
    minus_right = left_side - coeff*right_side
    minus_right = minus_right[:,:, 1:]
    minus_right[:,:, -1] = replace

    zips = torch.zeros((batch_size,max_i,1),dtype=torch.float).to(device)
    zips_and_across= torch.zeros((batch_size,1,max_j+1),dtype=torch.float).to(device)
    to_right=torch.cat([image,zips],axis=2)
    zip_se=torch.cat([to_right,zips_and_across],axis=1)
    to_left=torch.cat([zips,image],axis=2)
    zip_nw=torch.cat([zips_and_across,to_left],axis=1)
    minus_se = zip_nw - coeff * zip_se
    minus_se = minus_se[:,1:,1:]
    minus_se[:,-1,:]=replace
    minus_se[:,:,-1]=replace

    zips_ne = torch.cat([zips_and_across,to_right],axis=1)
    zips_sw = torch.cat([to_left,zips_and_across],axis=1)
    minus_sw = zips_ne - coeff*zips_sw
    minus_sw = minus_sw[:,1:,:-1]
    minus_sw[:,:,0] = replace
    minus_sw[:,-1,:] = replace
    return minus_right,minus_se,minus_below,minus_sw

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

if __name__=='__main__':
    image = torch.tensor([[1, 5, 9], [2, 6, 12], [7, 2, 1]],dtype=torch.float)
    right, se, below, sw = compute_diff(image)
    print(right, se, below, sw)