from layers.mnist_digit import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
from layers.sp_layer import SPLayer
import argparse
def train_image(image,lr,num_iters):
    '''
    Parameters
    ----------
    image: numpy.ndarray
     shape nxn
    lr: int
     learning rate
    num_iters: int
     training iterations
    Returns
    -------
    None
    '''
    plt.imshow(image)
    plt.show()
    writer=SummaryWriter()
    loc_to_idx, idx_to_loc, map, rev_map=make_graph(16,16)
    layer=SPLayer(idx_to_loc,map,rev_map)
    for i in range(num_iters):
        v,E,hard_v=layer.forward(image)
        image_grad=layer.backward(image, E, idx_to_loc)
        writer.add_scalar('soft shortest path',v,global_step=i)
        writer.add_scalar('hard shortest path',hard_v,global_step=i)

        if not i%10:
            grid = torchvision.utils.make_grid(image)
            writer.add_image('image', grid, global_step=i)
        if not i%100:
            plt.imshow(image)
            plt.show()
        image-=lr*image_grad

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float,default=01e-03)
    parser.add_argument("--n_iters",type=int,default=1000)

    args=parser.parse_args()

    image = get_cropped_image()

    train_image(image,args.lr,args.n_iters)
