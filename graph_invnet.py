import math
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from models.wgan import *
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
import os
from utils import calc_gradient_penalty,gen_rand_noise,\
                        weights_init,generate_image
from layers.sp_layer import SPLayer
from layers.graph_layer import GraphLayer
from layers.mnist_digit import make_graph
from config import *
import libs as lib
import libs.plot
import numpy as np
import sys
import time
from invnet import InvNet

class GraphInvNet(InvNet):

    def __init__(self,batch_size,output_path,data_dir,lr,critic_iters,\
                 proj_iters,output_dim,hidden_size,device,lambda_gp,max_i=32,max_j=32,restore_mode=False,sp_layer=SPLayer):

        super().__init__(batch_size,output_path,data_dir,lr,critic_iters,proj_iters,output_dim,hidden_size,device,lambda_gp,restore_mode)

        self.max_i,self.max_j=max_i,max_j
        _,_,self.adj_map,self.rev_map=make_graph(max_i,max_j)
        self.dp_layer=SPLayer.apply
        self.graph_layer=GraphLayer.apply
        self.start=timer()


    def proj_loss(self,fake_data,real_lengths):
        #TODO get numpy operations on gpu

        #TODO Try this without normalization
        fake_data = fake_data.view((self.batch_size, self.max_i, self.max_j))
        real_lengths=real_lengths.view(-1)
        thetas=self.graph_layer(fake_data)
        fake_lengths = self.dp_layer(thetas,self.adj_map,self.rev_map)
        proj_loss=F.mse_loss(fake_lengths,real_lengths)
        return proj_loss

    def real_p1(self,images):
        images=torch.squeeze(images)
        thetas=self.graph_layer(images)
        real_lengths=self.dp_layer(thetas,self.adj_map,self.rev_map)
        return real_lengths.view(-1,1)

    def sample(self,train=True):
        if train:
            try:
                real_data = next(self.dataiter)
            except:
                self.dataiter = iter(self.train_loader)
                real_data = self.dataiter.next()
            if real_data[0].shape[0]<self.batch_size:
                real_data=self.sample()
        else:
            try:
                real_data = next(self.val_iter)
            except:
                self.val_iter = iter(self.val_loader)
                real_data = self.val_iter.next()
            if real_data[0].shape[0]<self.batch_size:
                real_data=self.sample(train=False)
        return real_data

    def load_data(self):
        data_transform = transforms.Compose([
            transforms.Resize(32),
            # transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3801])
        ])
        data_dir = self.data_dir
        print('data_dir:',data_dir)
        mnist_data = datasets.MNIST(data_dir, download=True,
                                    transform=data_transform)
        train_data,val_data=torch.utils.data.random_split(mnist_data, [55000,5000])

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader=torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
        images, _ = next(iter(train_loader))
        return train_loader,val_loader

    def norm_data(self,data):
        avg = data.mean()
        std = data.std()

        normed = (data - avg) / (std / 0.8)
        return normed

if __name__=='__main__':
    config=InvNetConfig()


    cuda_available = torch.cuda.is_available()
    device = torch.device(config.gpu if cuda_available else "cpu")
    device="cuda:1"
    if cuda_available:
        torch.cuda.set_device(device)
    print('training on:',device)
    sys.stdout.flush()
    invnet=GraphInvNet(config.batch_size,config.output_path,config.data_dir,
                  config.lr,config.critic_iter,config.proj_iter,32*32,
                  config.hidden_size,device,config.lambda_gp)
    invnet.train(10000)
    #TODO fix proj_loss reporting
