import torch.nn.init as init
from models.wgan import *
from torch import autograd
from torchvision import transforms, datasets

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")


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

def calc_gradient_penalty(netD, real_data, fake_data,batch_size,lambd):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()

    alpha = alpha.view(batch_size, CATEGORY, DIM, DIM)
    alpha = alpha.to(device)

    fake_data = fake_data.view(batch_size, CATEGORY, DIM, DIM)
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

def gen_rand_noise(batch_size):
    noise = torch.randn(batch_size, 128)
    noise = noise.to(device)

    return noise


def load_data(batch_size):



    data_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    mnist_data=datasets.MNIST('/Users/kellymarshall/PycharmProjects/graph_invnet/files/',download=True,transform=data_transform)
    train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)
    return train_loader


def generate_image(netG, batch_size,conditional,noise=None, lv=None):
    if noise is None:
        noise = gen_rand_noise(batch_size)
    if conditional:
        if lv is None:
            # locationX and locationY randomly picks the centroid of the generated circles for the tensorboard.
            # radius is calculated based on the area of the circle.
            # using the conversion with (1/DIM)^2 * pi * r^2 = "normalized area",
            # 'r' is based on the unit of pixel.
            length=torch.rand(batch_size)*10
            lv = length
            lv = lv.to(device)
    else:
        lv = None
    with torch.no_grad():
        noisev = noise
        lv_v = lv
    samples = netG(noisev, lv_v)
    samples = torch.argmax(samples.view(batch_size, CATEGORY, DIM, DIM), dim=1).unsqueeze(1)
    samples = (samples * 255 / (CATEGORY))
    return samples


