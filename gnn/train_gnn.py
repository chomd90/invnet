import argparse
import torch
from torchvision import transforms,datasets
from microstructure import MicrostructureDataset
import torch
from layers import DPLayer
class GNNTrainer:

    def __init__(self,edge_fn,dataset,max_op):
        if dataset=='mnist':
            data_transform = transforms.Compose([
                transforms.Resize(self.max_i),
                # transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3801])
            ])
            data_dir = self.data_dir
            print('data_dir:', data_dir)
            mnist_data = datasets.MNIST(data_dir, download=True,
                                        transform=data_transform)
            train_data, val_data = torch.utils.data.random_split(mnist_data, [55000, 5000])

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
            images, _ = next(iter(train_loader))
            self.train_loader=train_loader
            self.val_loader=val_loader
            self.real_p1_layer = DPLayer(edge_fn, max_op, 32, 32, make_pos=True)

        if dataset=='2phase':
            data_dir='/data/datasets/two_phase_morph'
            train_dir = data_dir + 'morph_global_64_train_255.h5'
            test_dir = data_dir + 'morph_global_64_valid_255.h5'
            # Returns train_loader and val_loader, both of pytorch DataLoader type
            train_data = MicrostructureDataset(train_dir)
            test_data = MicrostructureDataset(test_dir)

            self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
            self.real_p1_layer=DPLayer(edge_fn,max_op,64,64,make_pos=False)


    def train_step(self):
        pass

def train(edge_fn):

    pass

if __name__=='__main__':
    device=torch.device('cuda:0')
    parser = argparse.ArgumentParser(description='gnn training')
    parser.add_argument('edge_fn', default='diff_squared')


    args = parser.parse_args()
    print(args.accumulate(args.integers))
