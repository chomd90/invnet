import torch

from layers.graph_layer.adjacency_utils import compute_diff


def test_compute_diff():
    cuda_available = torch.cuda.is_available()
    device = torch.device(0 if cuda_available else "cpu")

    image = torch.tensor([[1, 5, 9], [2, 6, 12], [7, 2, 1]],dtype=torch.float).to(device)
    image=image.unsqueeze(0)
    directions = list(compute_diff(image))

    one=torch.tensor([[ 6. ,14. , 0.], [ 8. ,18.  ,0.],[ 9.,  3. , 0.]]).to(device)
    two=torch.tensor([[ 7. ,17.,  0.],[ 4.,  7.,  0.], [ 0.,  0.,  0.]]).to(device)
    three= torch.tensor([[ 3. ,11., 21.],[ 9.,  8., 13.], [ 0.,  0.,  0.]]).to(device)
    four=torch.tensor([[ 0. , 7., 15.],[ 0., 13., 14.], [ 0.,  0.,  0.]]).to(device)

    true_directions=[one,two,three,four]
    for i,dir in enumerate(true_directions):
        true_directions[i]=dir.unsqueeze(0)
    for dir_idx,dir in enumerate(directions):
        for i in range(3):
            for j in range(3):
                assert true_directions[dir_idx][0,i,j]==dir[0,i,j]