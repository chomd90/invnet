import torch

from layers import DPLayer,edge_f_dict
from torchvision import transforms, datasets


def make_data():
    data_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3801])
    ])
    data_dir = '/data/MNIST'
    mnist_data = datasets.MNIST(data_dir, download=True,
                                transform=data_transform)
    train_data, val_data = torch.utils.data.random_split(mnist_data, [55000, 5000])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
    images, _ = next(iter(train_loader))
    train_loader = train_loader
    val_loader = val_loader
    real_p1_layer = DPLayer(edge_f_dict['diff_squared'], max_op, 32, 32, make_pos=True)