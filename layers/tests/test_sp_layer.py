import unittest
import torch
from layers.sp_layer import SPLayer
from layers.graph_layer import GraphLayer
from layers.sp_utils import adjacency_function
from layers.mnist_digit import make_graph

class TestShortestPath(unittest.TestCase):
    def test_forward(self):
        input_data=self.get_data().unsqueeze(0)
        other_input=input_data.clone()
        input_data=torch.cat([input_data,other_input],dim=0)
        input_data.requires_grad=True
        b,m_i,m_j=input_data.shape
        graph_layer=GraphLayer.apply
        sp_layer=SPLayer().apply
        thetas=graph_layer(input_data)
        _,_,adj,rev_adj=make_graph(m_i,m_j)
        lengths=sp_layer(thetas,adj,rev_adj)
        lengths.sum().backward()
        self.assertEqual(True, True)

    def get_data(self):
        a_range=torch.arange(1,4,dtype=torch.float).view((-1,1))
        return torch.matmul(a_range,a_range.transpose(0,1))




if __name__ == '__main__':
    unittest.main()
