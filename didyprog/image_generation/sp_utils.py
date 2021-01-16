import numpy as np
import torch

def compute_diff(image,add=True):
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

    zips= torch.zeros(image.shape[-1],dtype=torch.float).view(1,-1).to(device)
    zips_above=torch.cat([zips,image])
    zips_below=torch.cat([image,zips])
    minus_below = zips_above - coeff*zips_below
    minus_below = minus_below[1:, :]
    minus_below[-1, :] = 0

    zips = torch.zeros(image.shape[-1],dtype=torch.float).view(-1,1).to(device)
    right_side = torch.cat([image, zips], axis=1)
    left_side = torch.cat([zips, image], axis=1)
    minus_right = left_side - coeff*right_side
    minus_right = minus_right[:, 1:]
    minus_right[:, -1] = 0

    zips = torch.zeros(image.shape[-1],dtype=torch.float).view(-1,1).to(device)
    zips_and_across= torch.zeros(image.shape[-1]+1,dtype=torch.float).view(1,-1).to(device)
    to_right=torch.cat([image,zips],axis=1)
    zip_se=torch.cat([to_right,zips_and_across])
    to_left=torch.cat([zips,image],axis=1)
    zip_nw=torch.cat([zips_and_across,to_left])
    minus_se = zip_nw - coeff * zip_se
    minus_se = minus_se[1:,1:]
    minus_se[-1,:]=0
    minus_se[:,-1]=0

    zips_ne = torch.cat([zips_and_across,to_right])
    zips_sw = torch.cat([to_left,zips_and_across])
    minus_sw = zips_ne - coeff*zips_sw
    minus_sw = minus_sw[1:,:-1]
    minus_sw[:,0] = 0
    minus_sw[-1,:] = 0
    return minus_right,minus_se,minus_below,minus_sw

def idxloc(size_j,idx):
    return idx//size_j,idx%size_j

def locidx(size_j,idx_i,idx_j):
    return size_j*idx_i + idx_j

def adjacency(idx,max_i,max_j):
    i,j=idxloc(max_j,idx)
    if j<max_j-1:
        yield idx+1
        if i<max_i-1:
            yield idx+max_j+1
        else:
            yield None
    else:
        yield None
        yield None
    if i<max_i-1:
        yield idx+max_j
        if j>0:
            yield idx+max_j-1
        else:
            yield None
    else:
        yield None
        yield None


if __name__=='__main__':
    image = torch.tensor([[1, 5, 9], [2, 6, 12], [7, 2, 1]],dtype=torch.float)
    right, se, below, sw = compute_diff(image)
    print(right, se, below, sw)