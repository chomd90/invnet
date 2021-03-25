import torch.nn as nn
import h5py
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import autograd
from torch.utils.data import Dataset

from models.wgan import MyConvo2d


def weights_init(m):
    if isinstance(m, MyConvo2d):
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

def calc_gradient_penalty(netD, real_data, fake_data,batch_size,lambd,size):
    device=real_data.device

    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()

    alpha = alpha.view(batch_size, -1)
    alpha = alpha.to(device)

    fake_data = fake_data.view(batch_size,-1)
    real_data= real_data.view(batch_size,-1)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambd
    return gradient_penalty

class MicrostructureDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(MicrostructureDataset, self).__init__()
        self.data = h5py.File(data_path, mode='r')['morphology_64_64']
        self.transform = transform

    def __getitem__(self, index):
        x = torch.FloatTensor(self.data[index, ...])
        if self.transform is not None:
            x = self.transform(x)
        return x/255

    def __len__(self):
        return self.data.shape[0]
