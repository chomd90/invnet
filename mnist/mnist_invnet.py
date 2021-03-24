import sys

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets

from invnet import BaseInvNet
from mnist.config import *


class InvNet(BaseInvNet):

    def __init__(self,batch_size,output_path,data_dir,lr,critic_iters,\
                 proj_iters,hidden_size,device,lambda_gp,edge_fn,max_op,restore_mode=False):

        self.max_i,self.max_j=max_i,max_j
        self.max_op=max_op
        make_pos=True
        if edge_fn=='diff_exp':
            make_pos=False
        self.dp_layer=DPLayer(edge_fn,max_op,max_i,max_j,make_pos=make_pos)
        new_hparams = {'max_op': str(max_op), 'edge_fn': edge_fn}
        super().__init__(batch_size,output_path,data_dir,lr,critic_iters,proj_iters,max_i*max_j,hidden_size,device,lambda_gp,1,restore_mode,hparams=new_hparams)

    def proj_loss(self,fake_data,real_lengths):
        #TODO Experiment with normalization
        fake_data = fake_data.view((self.batch_size, self.max_i, self.max_j))
        real_lengths=real_lengths.view((-1,1))

        fake_lengths=self.real_p1(fake_data)
        proj_loss=F.mse_loss(fake_lengths,real_lengths)
        return proj_loss

    def real_p1(self,images):
        images=torch.squeeze(images)
        images=self.norm_data(images)
        real_lengths=self.dp_layer(images).view(-1,1)
        if self.p1_mean is not None:
            real_lengths=self.normalize_p1(real_lengths)
        return real_lengths

    def load_data(self):
        data_transform = transforms.Compose([
            transforms.Resize(self.max_i),
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
    config=Config()


    cuda_available = torch.cuda.is_available()
    device = torch.device(config.gpu if cuda_available else "cpu")
    device="cuda:0"
    if cuda_available:
        torch.cuda.set_device(device)
    print('training on:',device)
    sys.stdout.flush()

    invnet=InvNet(config.batch_size, config.output_path, config.data_dir,
                  config.lr, config.critic_iter, config.proj_iter,
                  config.hidden_size, device, config.lambda_gp, config.edge_fn, config.max_op)
    invnet.train(10000)
    #TODO fix proj_loss reporting
