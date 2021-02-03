import math
import sys

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets

from config import *
from invnet import BaseInvNet
from layers.DPLayer import DPLayer


class GraphInvNet(BaseInvNet):

    def __init__(self,batch_size,output_path,data_dir,lr,critic_iters,\
                 proj_iters,hidden_size,device,lambda_gp,edge_fn,max_op,max_i=32,max_j=32,restore_mode=False):

        self.max_i,self.max_j=max_i,max_j
        self.max_op=max_op
        self.dp_layer=DPLayer(edge_fn,max_op,max_i,max_j)
        new_hparams = {'max_op': str(max_op), 'edge_fn': edge_fn}
        super().__init__(batch_size,output_path,data_dir,lr,critic_iters,proj_iters,max_i*max_j,hidden_size,device,lambda_gp,1,restore_mode,hparams=new_hparams)

    def proj_loss(self,fake_data,real_lengths):
        #TODO Experiment with normalization
        fake_data = fake_data.view((self.batch_size, self.max_i, self.max_j))
        real_lengths=real_lengths.view(-1)

        fake_lengths=self.dp_layer(fake_data)
        proj_loss=F.mse_loss(fake_lengths,real_lengths)
        return proj_loss

    def real_p1(self,images):
        images=torch.squeeze(images)
        images=self.norm_data(images)
        real_lengths=self.dp_layer(images)
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
        return real_data[0]

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

    def save(self,stats):
        #TODO split this into base saving actions and MNIST/DP specific saving stuff
        size = int(math.sqrt(self.output_dim))
        fake_2 = stats['fake_data'].view(self.batch_size, -1, size, size)
        fake_2 = fake_2.int()
        fake_2 = fake_2.cpu().detach().clone()
        fake_2 = torchvision.utils.make_grid(fake_2, nrow=8, padding=2)
        self.writer.add_image('G/images', fake_2, stats['iteration'])

        dev_proj_err, dev_disc_cost=self.validate()
        #Generating images for tensorboard display
        lv=torch.tensor([600,780,960,1140]).view(-1,1).float().to(device)
        with torch.no_grad():
            noisev=self.fixed_noise
            lv_v=lv
        noisev=noisev.float()
        gen_images=self.G(noisev,lv_v).view((self.batch_size,-11,size,size))
        gen_images = self.norm_data(gen_images)
        real_images = stats['real_data']
        real_grid_images = torchvision.utils.make_grid(real_images[:4], nrow=8, padding=2)
        fake_grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
        real_grid_images = real_grid_images.long()
        fake_grid_images = fake_grid_images.long()
        self.writer.add_image('real images', real_grid_images, stats['iteration'])
        self.writer.add_image('fake images', fake_grid_images, stats['iteration'])
        torch.save(self.G, self.output_path + 'generator.pt')
        torch.save(self.D, self.output_path + 'discriminator.pt')


        metric_dict = {'generator_cost': stats['gen_cost'],
                       'discriminator_cost': dev_disc_cost ,'validation_projection_error': dev_proj_err}
        self.writer.add_hparams(self.hparams, metric_dict,global_step=stats['iteration'])

if __name__=='__main__':
    config=InvNetConfig()


    cuda_available = torch.cuda.is_available()
    device = torch.device(config.gpu if cuda_available else "cpu")
    device="cuda:0"
    if cuda_available:
        torch.cuda.set_device(device)
    print('training on:',device)
    sys.stdout.flush()

    invnet=GraphInvNet(config.batch_size,config.output_path,config.data_dir,
                  config.lr,config.critic_iter,config.proj_iter,
                  config.hidden_size,device,config.lambda_gp,config.edge_fn,config.max_op)
    invnet.train(10000)
    #TODO fix proj_loss reporting
