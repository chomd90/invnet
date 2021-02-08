import torch.nn as nn

from layers.dp_layer.adjacency_utils import idx_adjacency
from layers.graph_layer import GraphLayer
from layers.dp_layer.dp_function import DPFunction
from layers.dp_layer.edge_functions import edge_f_dict

class DPLayer(nn.Module):

    def __init__(self,edge_fn,max_op,max_i,max_j,make_pos=True):
        super(DPLayer, self).__init__()
        self.edge_f=edge_f_dict[edge_fn]
        self.max_op=max_op
        null = float('inf')
        if self.max_op:
            null *= -1
        self.graph_layer = GraphLayer(null,self.edge_f,make_pos)
        self.sp_function=DPFunction.apply
        self.adj_array,self.rev_adj=idx_adjacency(max_i,max_j)

    def forward(self,images):
        thetas = self.graph_layer(images)
        fake_lengths = self.sp_function(thetas, self.adj_map, self.rev_map,self.max_op)
        return fake_lengths

