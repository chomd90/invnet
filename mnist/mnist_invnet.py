import sys

import torch

from invnet import BaseInvNet
from mnist.config import *


class InvNet(BaseInvNet):

    def __init__(self,batch_size,output_path,data_dir,lr,critic_iters,\
                 proj_iters,hidden_size,device,lambda_gp,edge_fn,max_op,restore_mode=False):

        self.max_op=max_op
        make_pos=True
        if edge_fn=='diff_exp':
            make_pos=False
        new_hparams = {'max_op': str(max_op), 'edge_fn': edge_fn}
        super().__init__(batch_size,output_path,data_dir,lr,critic_iters,proj_iters,32,32,hidden_size,device,lambda_gp,1,restore_mode,edge_fn,max_op,make_pos,hparams=new_hparams)

    def real_p1(self,images):
        images=torch.squeeze(images)
        images=self.norm_data(images)
        real_lengths=self.dp_layer(images).view(-1,1)
        if self.p1_mean is not None:
            real_lengths=self.normalize_p1(real_lengths)
        return real_lengths

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
