import torch.nn as nn
from layers.graph_layer import GraphLayer
from layers.sp_function import SPFunction
from layers.make_graph import make_graph
from layers.edge_functions import edge_f_dict

class DPLayer(nn.Module):

    def __init__(self,edge_fn,max_op,max_i,max_j):
        super(DPLayer, self).__init__()
        self.edge_f=edge_f_dict[edge_fn]
        self.max_op=max_op
        null = float('inf')
        if self.max_op:
            null *= -1
        self.graph_layer = GraphLayer(null=null, edge_f=self.edge_f)
        self.sp_function=SPFunction.apply
        _,_,self.adj_map,self.rev_map=make_graph(max_i,max_j)

    def forward(self,images):
        thetas = self.graph_layer(images)
        fake_lengths = self.sp_function(thetas, self.adj_map, self.rev_map,self.max_op)
        return fake_lengths

