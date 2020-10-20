import torch, torchvision
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from queue import Queue
from collections import defaultdict

random.seed(0)
np.random.seed(0)
def crop(image,size):
    image_width, image_height = image.size()
    crop_height, crop_width = size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return image[crop_top:crop_top+crop_height,crop_left:crop_left+crop_width]



def get_cropped_image():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/Users/kellymarshall/PycharmProjects/didyprog/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()])),
        batch_size=1, shuffle=True)
    _, (example_data, example_targets) = next(enumerate(train_loader))
    image=example_data[0][0]
    cropped = crop(image, [16, 16])
    return cropped


def make_graph(image):
    max_i,max_j=image.shape
    idx_to_loc=[]
    loc_to_idx={}
    total_nodes=0


    reachable={}
    for i in range(max_i):
        for j in range(max_j):
            pixel=image[i][j]
            if pixel==0 or ((i,j) not in reachable and total_nodes>0):
                continue
            if total_nodes==0:
                reachable[(i,j)]=1
            idx_to_loc.append((i,j))
            loc_to_idx[(i,j)]=total_nodes
            reachable[(i+1,j)]=1
            reachable[(i,j+1)]=1
            total_nodes+=1
    theta=np.zeros(shape=(total_nodes,2))
    map={}
    rev_map=defaultdict(lambda: [None,None])
    #first row is downward distance,
    #second row rightward distance
    for i in range(total_nodes):
        i_loc,j_loc=idx_to_loc[i]
        value_i=image[i_loc,j_loc]
        if i_loc<=max_i-1 and (i_loc+1,j_loc) in loc_to_idx:
            down_val = (image[i_loc + 1, j_loc])
            down_idx=(i_loc+1,j_loc)
            down_i=loc_to_idx[down_idx]
            distance = (value_i - down_val) ** 2
            theta[i][0]=distance
        else:
            down_i=None
            theta[i][0]=float('inf')
        if j_loc<=max_j-1 and (i_loc,j_loc+1) in loc_to_idx:
            right_val = (image[i_loc, j_loc+1])
            right_idx = (i_loc, j_loc+1)
            right_i = loc_to_idx[right_idx]
            distance = (value_i - right_val) ** 2
            theta[i][1] = distance
        else:
            right_i=None
            theta[i][1]=float('inf')
        map[i]=[down_i,right_i]
        rev_map[down_i][0]=i
        rev_map[right_i][1]=i
    return theta,loc_to_idx,idx_to_loc,map,rev_map

def get_image_graph():
    image=get_cropped_image()
    edges,loc_to_idx,idx_to_loc=make_graph(image)
    return edges,loc_to_idx,idx_to_loc


def prune_graph(edges,loc_to_idx,idx_to_loc):
    visited_idxs,max_idx=get_max_idx(loc_to_idx,idx_to_loc)
    edges=edges[:max_idx,:max_idx]
    removable_locs=[]
    for i,loc in enumerate(idx_to_loc):
        if i>max_idx:
            removable_locs.append(loc)
    for loc in removable_locs:
        del loc_to_idx[loc]
        idx_to_loc.remove(loc)
    return edges,loc_to_idx,idx_to_loc

def get_max_idx(loc_to_idx,idx_to_loc):
    visited_idxs={}
    grays=[0]
    max_idx=0
    while len(grays)>0:
        idx=grays.pop()
        i,j=idx_to_loc[idx]
        candidates=[(i+1,j),(i,j+1)]
        for cand in candidates:
            if cand in loc_to_idx:
                cand_idx=loc_to_idx[cand]
                if cand_idx not in visited_idxs and cand_idx not in grays:
                    max_idx=max(cand_idx,max_idx)
                    grays.append(cand_idx)
        visited_idxs[idx]=True
    return visited_idxs,max_idx

def get_pruned_graph():
    image = get_cropped_image()
    edges, loc_to_idx, idx_to_loc = make_graph(image)
    edges, loc_to_idx, idx_to_loc =prune_graph(edges, loc_to_idx, idx_to_loc)
    return edges, loc_to_idx, idx_to_loc


if __name__=='__main__':
    image=get_cropped_image()
    edges,loc_to_idx,idx_to_loc=make_graph(image)
    p_edges,p_loc,p_idx=prune_graph(edges,loc_to_idx,idx_to_loc)
    plt.imshow(image)
    plt.show()

#TODO adapt graph-pruning code (and methods in this file generally) to work with adjacency map method