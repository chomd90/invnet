import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import defaultdict
from torchvision import transforms,datasets

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
        datasets.MNIST('/Users/kellymarshall/PycharmProjects/didyprog/files/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                   ])),
        batch_size=1, shuffle=True)
    _, (example_data, example_targets) = next(enumerate(train_loader))
    image=example_data[0][0]
    cropped = crop(image, [16, 16])
    return cropped


def make_graph(max_i,max_j):
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

if __name__=='__main__':
    image=get_cropped_image()
    edges,loc_to_idx,idx_to_loc=make_graph(image)
    plt.imshow(image)
    plt.show()

#TODO adapt graph-pruning code (and methods in this file generally) to work with adjacency map method