import torch.nn.functional as F
import torch
from torchvision import transforms, datasets
from layers.DPLayer import DPLayer
from config import *
import sys
from invnet import BaseInvNet

class GraphInvNet(BaseInvNet):

    def __init__(self,batch_size,output_path,data_dir,lr,critic_iters,\
                 proj_iters,hidden_size,device,lambda_gp,edge_fn,max_op,max_i=32,max_j=32,restore_mode=False):

        self.max_i,self.max_j=max_i,max_j
        self.max_op=max_op
        self.dp_layer=DPLayer(edge_fn,max_op,max_i,max_j)
        new_hparams = {'max_op': str(max_op), 'edge_fn': edge_fn}
        super().__init__(batch_size,output_path,data_dir,lr,critic_iters,proj_iters,max_i*max_j,hidden_size,device,lambda_gp,restore_mode,hparams=new_hparams)



    def proj_loss(self,fake_data,real_lengths):
        #TODO Experiment with normalization
        fake_data = fake_data.view((self.batch_size, self.max_i, self.max_j))
        real_lengths=real_lengths.view(-1)
        fake_lengths=self.dp_layer(fake_data)
        proj_loss=F.mse_loss(fake_lengths,real_lengths)
        return proj_loss

    def real_p1(self,images):
        images=torch.squeeze(images)
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
        return real_data

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
    config=InvNetConfig()


    cuda_available = torch.cuda.is_available()
    device = torch.device(config.gpu if cuda_available else "cpu")
    device="cuda:1"
    if cuda_available:
        torch.cuda.set_device(device)
    print('training on:',device)
    sys.stdout.flush()

    invnet=GraphInvNet(config.batch_size,config.output_path,config.data_dir,
                  config.lr,config.critic_iter,config.proj_iter,
                  config.hidden_size,device,config.lambda_gp,config.edge_fn,config.max_op)
    invnet.train(5000)
    #TODO fix proj_loss reporting
