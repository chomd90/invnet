from invnet import BaseInvNet
import torch
from micro_config import MicroConfig
from microstructure_dataset import MicrostructureDataset
from layers import DPLayer
import torch.nn.functional as F
class MicroInvnet(BaseInvNet):

    def __init__(self,batch_size,output_path,data_dir,lr,critic_iters,\
                 proj_iters,hidden_size,device,lambda_gp,edge_fn,max_op,restore_mode=False):

        DIM=20
        super().__init__(batch_size,output_path,data_dir,lr,critic_iters,proj_iters,64**2,hidden_size,device,lambda_gp,restore_mode)

        self.edge_fn=edge_fn
        self.max_op=max_op
        # self.DPLayer=DPLayer('edge')

    def proj_loss(self,fake_data,real_p1):
        pass

    def real_p1(self,images):
        #For now let's do max_path over total size
        pass

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
        output=F.one_hot(idxs,num_classes=6)
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
        pass

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