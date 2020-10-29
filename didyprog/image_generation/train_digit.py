from image_generation.mnist_digit import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
from image_generation.sp_layer import SPLayer
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
    lengths=[]
    writer=SummaryWriter()
    layer=SPLayer()
    for i in range(num_iters):
        v,E,idx_to_loc=layer.forward(image)
        image_grad=layer.backward(image, E, idx_to_loc)
        hard_v=layer.true_value(image)
        lengths.append(v)

        writer.add_scalar('soft shortest path',v,global_step=i)
        writer.add_scalar('hard shortest path',hard_v,global_step=i)

        if not i%10:
            grid = torchvision.utils.make_grid(image)
            writer.add_image('image', grid, global_step=i)
        if not i%100:
            plt.imshow(image)
            plt.show()
        image+=lr*image_grad

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float,default=01e-03)
    parser.add_argument("--n_iters",type=int,default=1000)

    args=parser.parse_args()

    image = get_cropped_image()

    print(type(args.n_iters))
    train_image(image,args.lr,args.n_iters)
