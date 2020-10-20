import os, sys
import time
import libs as lib
import libs.plot
from tensorboardX import SummaryWriter
from models.wgan import *
from models.checkers import *
from config import InvNetConfig
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from timeit import default_timer as timer
from matscidata import MatSciDataset
from fixedcircledata import CircleDataset
import torch.nn.init as init
from graph.utils import weights_init,calc_gradient_penalty,gen_rand_noise,load_data,generate_image,training_data_loader,val_data_loader,proj_loss
from image_generation.sp_layer import SPLayer

layer=SPLayer()
config = InvNetConfig()

DATA_DIR = config.trainset_path
VAL_DIR = config.validset_path

IMAGE_DATA_SET = config.dataset
# torch.cuda.set_device(config.gpu)

if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory!')

NUM_CIRCLE = 0
if config.dataset == 'circle':
    NUM_CIRCLE = 2
    CATEGORY = NUM_CIRCLE + 1
    CNTL_DIM = CATEGORY + NUM_CIRCLE * 2
    CONDITIONAL = True

if NUM_CIRCLE == 0 and config.dataset == 'circle':
    raise Exception('Dataset is circle but NUM_CIRCLE == 0.')

RESTORE_MODE = config.restore_mode  # if True, it will load saved model from OUT_PATH and continue to train
START_ITER = 0  # starting iteration
OUTPUT_PATH = config.output_path

CRITIC_ITERS = config.critic_iter  # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 1  # Number of GPUs
BATCH_SIZE = config.batch_size

END_ITER = config.end_iter  # How many iterations to train for
LAMBDA = config.lambda_gp  # Gradient penalty lambda hyperparameter
OUTPUT_DIM = DIM * DIM  # Number of pixels in each image
PJ_ITERS = config.proj_iter

sp_layer=SPLayer()

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
fixed_noise = gen_rand_noise(BATCH_SIZE)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

if RESTORE_MODE:
    aG = torch.load(OUTPUT_PATH + "generator.pt")
    aD = torch.load(OUTPUT_PATH + "discriminator.pt")
else:
    print('DIM and DIMMER:',DIM*DIM)
    if CONDITIONAL:
        aG = GoodGenerator(64, int(DIM * DIM), ctrl_dim=10)  # +4 for the centroid.
        aD = GoodDiscriminator(64)
    else:
        aG = GoodGenerator(64, int(DIM * DIM), ctrl_dim=10)
        aD = GoodDiscriminator(64)
aG.apply(weights_init)
aD.apply(weights_init)

optimizer_g = torch.optim.Adam(aG.parameters(), lr=config.lr, betas=(0, 0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=config.lr, betas=(0, 0.9))
if CONDITIONAL:
    optimizer_pj = torch.optim.Adam(aG.parameters(), lr=config.lr, betas=(0, 0.9))
one = torch.FloatTensor([1])
mone = one * -1
aG = aG.to(device)
aD = aD.to(device)
one = one.to(device)
mone = mone.to(device)

writer = SummaryWriter()

def generator_update(aG, aD, real_data,optimizer_g):
    if CONDITIONAL:
        real_class=F.one_hot(torch.tensor(real_data[1]),num_classes=10)
        real_class=real_class.float()
        real_p1 = real_class.to(device)
    else:
        real_p1 = None
    for i in range(GENER_ITERS):
        print("Generator iters: " + str(i))
        aG.zero_grad()
        noise = gen_rand_noise(BATCH_SIZE)
        noise.requires_grad_(True)
        fake_data = aG(noise, real_p1)
        gen_cost = aD(fake_data)
        gen_cost = gen_cost.mean()
        gen_cost = gen_cost.view((1))
        gen_cost.backward(mone)
        gen_cost = -gen_cost

    optimizer_g.step()

    return gen_cost,real_p1

def projection_update(aG,real_data,real_p1,optimizer_pj):
    pj_cost = None
    for i in range(PJ_ITERS):
        print('Projection iters: {}'.format(i))
        aG.zero_grad()
        noise = gen_rand_noise(BATCH_SIZE)
        noise.requires_grad = True
        fake_data = aG(noise, real_p1)
        pj_cost = proj_loss(fake_data.view(-1, CATEGORY, DIM, DIM), real_data.to(device))
        pj_cost = pj_cost.mean()
        pj_cost.backward()
        optimizer_pj.step()
    return pj_cost


def critic_update(aG,aD,optimizer_d,real_data,iteration):
    for p in aD.parameters():  # reset requires_grad
        p.requires_grad_(True)  # they are set to False below in training G
    for i in range(CRITIC_ITERS):
        print("Critic iter: " + str(i))

        start = timer()
        aD.zero_grad()

        # gen fake data and load real data
        noise = gen_rand_noise(BATCH_SIZE)

        # batch = batch[0] #batch[1] contains labels
        real_images = real_data[0].to(device)  # TODO: modify load_data for each loading
        # real_p1.to(device)
        with torch.no_grad():
            noisev = noise  # totally freeze G, training D
            if CONDITIONAL:
                real_class = F.one_hot(torch.tensor(real_data[1]), num_classes=10)
                real_class = real_class.float()
                real_p1 = real_class.to(device)
            else:
                real_p1 = None
        end = timer()
        print(f'---gen G elapsed time: {end - start}')
        start = timer()
        fake_data = aG(noisev, real_p1).detach()
        end = timer()
        print(f'---load real imgs elapsed time: {end - start}')
        start = timer()

        # train with real data
        disc_real = aD(real_images)
        disc_real = disc_real.mean()

        # train with fake data
        disc_fake = aD(fake_data)
        disc_fake = disc_fake.mean()

        # train with interpolates data
        gradient_penalty = calc_gradient_penalty(aD, real_images, fake_data, BATCH_SIZE, LAMBDA)

        # final disc cost
        disc_cost = disc_fake - disc_real + gradient_penalty
        disc_cost.backward()
        w_dist = disc_fake - disc_real

        optimizer_d.step()
        if i == CRITIC_ITERS - 1:
            # ------------------VISUALIZATION----------
            writer.add_scalar('data/disc_cost', disc_cost, iteration)
            writer.add_scalar('data/disc_fake', disc_fake, iteration)
            writer.add_scalar('data/disc_real', disc_real, iteration)
            writer.add_scalar('data/gradient_pen', gradient_penalty, iteration)
        end = timer();
        print(f'---train D elapsed time: {end - start}')
    return w_dist,disc_cost


def sample(dataiter, dataloader):
    try:
        real_data = next(dataiter)
    except:
        dataiter = iter(dataloader)
        real_data = dataiter.next()
    return real_data, dataiter

# Reference: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
def train():
    dataloader = training_data_loader()
    dataiter = iter(dataloader)
    print(dataiter)
    for iteration in range(START_ITER, END_ITER):
        start_time = time.time()
        print("Iter: " + str(iteration))
        start = timer()
        # ---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D

        gen_cost = None
        real_data,dataiter=sample(dataiter,dataloader)
        gen_cost,real_p1=generator_update(aG, aD, real_data,optimizer_g)

        end = timer()
        # if CONDITIONAL:
        #     # Projection steps
        #     pj_cost=projection_update(aG,real_data,real_p1,optimizer_pj)
        #     writer.add_scalar('data/p1_cost', pj_cost.cpu().detach(), iteration)
        # ---------------------TRAIN D------------------------
        batch = next(dataiter, None)
        if batch is None:
            dataiter = iter(dataloader)
            batch = dataiter.next()
        w_dist,disc_cost=critic_update(aG,aD,optimizer_d,batch,iteration)

        # ---------------VISUALIZATION---------------------
        writer.add_scalar('data/gen_cost', gen_cost, iteration)

        lib.plot.plot(OUTPUT_PATH + 'time', time.time() - start_time)
        lib.plot.plot(OUTPUT_PATH + 'train_disc_cost', disc_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + 'train_gen_cost', gen_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + 'wasserstein_distance', w_dist.cpu().data.numpy())

        if iteration % 10 == 0:
            fake_2 = torch.argmax(fake_data.view(BATCH_SIZE, CATEGORY, DIM, DIM), dim=1).unsqueeze(1)
            fake_2 = (fake_2 * 255 / (CATEGORY))
            fake_2 = fake_2.int()
            fake_2 = fake_2.cpu().detach().clone()
            fake_2 = torchvision.utils.make_grid(fake_2, nrow=8, padding=2)
            writer.add_image('G/images', fake_2, iteration)

            val_loader = val_data_loader()
            dev_disc_costs = []
            for _, images in enumerate(val_loader):
                imgs = torch.Tensor(images[0])
                imgs = imgs.to(device)
                with torch.no_grad():
                    imgs_v = imgs

                D = aD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot(OUTPUT_PATH + 'dev_disc_cost.png', np.mean(dev_disc_costs))
            lib.plot.flush()
            gen_images = generate_image(aG, fixed_noise)
            torchvision.utils.save_image(gen_images, OUTPUT_PATH + 'samples_{}.png'.format(iteration), nrow=8,
                                         padding=2)
            grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
            writer.add_image('images', grid_images, iteration)
            # ----------------------Save model----------------------
            torch.save(aG, OUTPUT_PATH + "generator.pt")
            torch.save(aD, OUTPUT_PATH + "discriminator.pt")
        lib.plot.tick()


train()


