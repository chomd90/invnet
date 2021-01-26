from invnet import BaseInvNet
import torch
from micro_config import MicroConfig
from microstructure_dataset import MicrostructureDataset
from layers import DPLayer
import torch.nn.functional as F
import torch.nn as nn
from models.wgan import MicrostructureGenerator,GoodDiscriminator

class MicroInvnet(BaseInvNet):

    def __init__(self,batch_size,output_path,data_dir,lr,critic_iters,\
                 proj_iters,hidden_size,device,lambda_gp,edge_fn,max_op,restore_mode=False):

        super().__init__(batch_size,output_path,data_dir,lr,critic_iters,proj_iters,64**2,hidden_size,device,lambda_gp,6,6,restore_mode)

        self.G=MicrostructureGenerator(32, 6*64*64, ctrl_dim=6).to(device)
        self.D=GoodDiscriminator(6).to(device)
        self.edge_fn=edge_fn
        self.max_op=max_op
        self.DPLayer=DPLayer('v1_only',max_op,64,64,make_pos=False)

    def proj_loss(self,fake_data,real_p1):
        fake_lengths=self.real_p1(fake_data)
        loss_f=nn.MSELoss()
        return loss_f(fake_lengths,real_p1)

    def real_p1(self,images):
        '''
        Images: [b,64,64,6]
        p1: [b,6]
        '''
        rot_1=torch.transpose(images,2,3)
        rot_2 = torch.transpose(rot_1, 1, 2)
        input=torch.flatten(rot_2,0,1)
        lengths=self.DPLayer(input)
        return lengths.view((self.batch_size,-1))

    def sample(self,train=True):
        if train:
            try:
                real_data = next(self.dataiter)
            except:
                self.dataiter = iter(self.train_loader)
                real_data = self.dataiter.next()
            if real_data[0].shape[0] < self.batch_size:
                real_data = self.sample()
        else:
            try:
                real_data = next(self.val_iter)
            except:
                self.val_iter = iter(self.val_loader)
                real_data = self.val_iter.next()
            if real_data[0].shape[0] < self.batch_size:
                real_data = self.sample(train=False)
        idxs=real_data.long()-1
        output=F.one_hot(idxs,num_classes=6).float()
        return output

    def load_data(self):
        train_dir = self.data_dir + '/train/train_30000_lhs.h5'
        test_dir = self.data_dir + '/test/valid_6000_lhs.h5'
        #Returns train_loader and val_loader, both of pytorch DataLoader type
        train_data=MicrostructureDataset(train_dir)
        test_data=MicrostructureDataset(test_dir)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        return train_loader,test_loader

    def norm_data(self,data):
        return data

    def format_data(self,data):
        return data.view((self.batch_size,64,64,6))

if __name__=="__main__":

    config = MicroConfig()

    cuda_available = torch.cuda.is_available()
    device = torch.device(config.gpu if cuda_available else "cpu")
    device = "cuda:1"
    if cuda_available:
        torch.cuda.set_device(device)
    print('training on:', device)

    invnet = MicroInvnet(config.batch_size, config.output_path, config.data_dir,
                         config.lr, config.critic_iter, config.proj_iter,
                         config.hidden_size, device, config.lambda_gp, config.edge_fn, config.max_op)
    invnet.train(5000)
    #