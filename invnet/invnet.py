import math
import os
import time
from abc import ABC, abstractmethod
from timeit import default_timer as timer

import numpy as np
import torchvision
from tensorboardX import SummaryWriter

from invnet import calc_gradient_penalty, \
    weights_init
from layers import DPLayer
from models.wgan import *


class BaseInvNet(ABC):

    def __init__(self, batch_size, output_path, data_dir, lr, critic_iters, proj_iters, output_dim, hidden_size, device, lambda_gp,ctrl_dim,restore_mode=False,hparams={}):
        self.writer = SummaryWriter()
        print('output dir:', self.writer.logdir)
        self.device = device

        self.data_dir = data_dir
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.batch_size = batch_size
        self.output_dim = output_dim
        self.lambda_gp = lambda_gp

        self.train_loader, self.val_loader = self.load_data()
        self.dataiter, self.val_iter = iter(self.train_loader), iter(self.val_loader)

        self.critic_iters = critic_iters
        self.proj_iters = proj_iters
        hparams.update({'proj_iters': self.proj_iters,
                      'critic_iters': self.critic_iters})
        self.hparams=hparams


        if restore_mode:
            self.D = torch.load(output_path + "generator.pt").to(device)
            self.G = torch.load(output_path + "discriminator.pt").to(device)
        else:
            self.G = GoodGenerator(hidden_size, self.output_dim, ctrl_dim=ctrl_dim).to(device)
            self.D = GoodDiscriminator(dim=hidden_size).to(device)
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        self.optim_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0, 0.9))
        self.optim_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0, 0.9))
        self.optim_pj = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0, 0.9))

        self.fixed_noise = self.gen_rand_noise(4)
        self.p1_mean, self.p1_std = None,None
        self.p1_mean, self.p1_std = self.get_p1_stats()

        self.start = timer()

    def train(self, iters):

        for iteration in range(iters):
            print('iteration:', iteration)

            gen_cost, real_p1 = self.generator_update()
            start_time = time.time()
            proj_cost = self.proj_update()
            stats = self.critic_update()
            add_stats = {'start': start_time,
                         'iteration': iteration,
                         'gen_cost': gen_cost,
                         'proj_cost': proj_cost}
            stats.update(add_stats)
            if iteration%5==0:
                stats['val_proj_err'], stats['val_critic_err'] = self.validation()
                self.log(stats)
            if iteration % 50 == 0:
                self.save(stats)

    def generator_update(self):
        start=timer()
        for p in self.D.parameters():
            p.requires_grad_(False)

        real_data= self.sample()
        real_images=real_data.to(self.device)
        with torch.no_grad():
            real_lengths=self.real_p1(real_images)
        real_p1=real_lengths.to(self.device)
        mone = torch.FloatTensor([1]) * -1
        mone=mone.to(self.device)

        for i in range(1):
            self.G.zero_grad()
            noise = self.gen_rand_noise(self.batch_size).to(self.device)
            noise.requires_grad_(True)
            fake_data = self.G(noise, real_p1)
            fake_data = self.format_data(fake_data)
            gen_cost = self.D(fake_data)
            gen_cost = gen_cost.mean()
            gen_cost = gen_cost.view((1))
            gen_cost.backward(mone)
            gen_cost = -gen_cost

            self.optim_g.step()

        end=timer()
        # print('--generator update elapsed time:',end-start)
        return gen_cost, real_p1

    def critic_update(self):
        for p in self.D.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        start = timer()
        for i in range(self.critic_iters):
            self.D.zero_grad()
            real_images = self.sample().to(self.device)
            # gen fake data and load real data
            noise = self.gen_rand_noise(self.batch_size).to(self.device)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
                real_lengths= self.real_p1(real_images)
                real_p1 = real_lengths.to(self.device)
            fake_data = self.G(noisev, real_p1).detach()
            # train with real data
            disc_real = self.D(real_images)
            disc_real = disc_real.mean()

            # train with fake data
            disc_fake = self.D(fake_data)
            disc_fake = disc_fake.mean()

            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(self.D, real_images, fake_data, self.batch_size, self.lambda_gp,int(math.sqrt(self.output_dim)))

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward()
            w_dist = disc_fake - disc_real

            self.optim_d.step()
        end = timer()
        # print('---train D elapsed time:', end - start)
        stats={'w_dist': w_dist,
               'disc_cost':disc_cost,
               'fake_data':fake_data[:100],
               'real_data':real_images[:4],
               'disc_real':disc_real,
               'disc_fake':disc_fake,
               'gradient_penalty':gradient_penalty,
               'real_p1_avg':real_p1.mean(),
                'real_p1_std':real_p1.std()}
        return stats

    def proj_update(self):
        if not self.proj_iters:
            return 0
        start=timer()
        real_data = self.sample()
        total_pj_loss=torch.tensor([0.])
        with torch.no_grad():
            images = real_data.to(self.device)
            real_lengths = self.real_p1(images).view(-1, 1)
        for iteration in range(self.proj_iters):
            self.G.zero_grad()
            noise=self.gen_rand_noise(self.batch_size).to(self.device)
            noise.requires_grad=True
            fake_data = self.G(noise, real_lengths).view((self.batch_size,self.max_i,self.max_j))
            pj_loss=self.proj_loss(fake_data,real_lengths)
            pj_loss.backward()
            total_pj_loss+=pj_loss.cpu()
            self.optim_pj.step()

        end=timer()
        # print('--projection update elapsed time:',end-start)
        return total_pj_loss/self.proj_iters

    def validation(self):
        proj_errors = []
        dev_disc_costs = []
        for batch in range(3):
            images=self.sample(train=False)
            if isinstance(images,list):
                images=images[0]
            imgs = torch.Tensor(images)
            imgs = imgs.to(self.device).squeeze()
            with torch.no_grad():
                imgs_v = imgs
                real_lengths = self.real_p1(imgs_v)
                noise = self.gen_rand_noise(real_lengths.shape[0]).to(self.device)
                fake_data = self.G(noise, real_lengths.to(self.device)).detach()
                _proj_err = self.proj_loss(fake_data, real_lengths).detach()
                D = self.D(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
            proj_errors.append(_proj_err)
            dev_disc_costs.append(_dev_disc_cost)
        dev_disc_cost = np.mean(dev_disc_costs)
        proj_error = sum(proj_errors)/(len(proj_errors)*self.batch_size)
        return proj_error, dev_disc_cost

    def log(self,stats):
        # ------------------VISUALIZATION----------
        self.writer.add_scalar('data/gen_cost', stats['gen_cost'], stats['iteration'])
        self.writer.add_scalar('data/disc_cost', stats['disc_cost'], stats['iteration'])
        self.writer.add_scalar('data/disc_fake', stats['disc_fake'], stats['iteration'])
        self.writer.add_scalar('data/disc_real', stats['disc_real'], stats['iteration'])
        self.writer.add_scalar('data/gradient_pen', stats['gradient_penalty'], stats['iteration'])
        self.writer.add_scalar('data/proj_error',stats['val_proj_err'],stats['iteration'])

    def gen_rand_noise(self,batch_size=None):
        if batch_size is None:
            batch_size=self.batch_size
        noise = torch.randn((batch_size, 128))
        noise = noise.to(self.device)
        return noise

    def format_data(self,data):
        return data

    def sample(self,train=True):
        if train:
            try:
                real_data = next(self.dataiter)
            except:
                self.dataiter = iter(self.train_loader)
                real_data = self.dataiter.next()
            if isinstance(real_data, list):
                real_data = real_data[0]

            if real_data.shape[0] < self.batch_size:
                real_data = self.sample()
        else:
            try:
                real_data = next(self.val_iter)
            except:
                self.val_iter = iter(self.val_loader)
                real_data = self.val_iter.next()
            if isinstance(real_data, list):
                real_data = real_data[0]
            if real_data.shape[0] < self.batch_size:
                real_data = self.sample(train=False)
        return real_data.squeeze()

    def get_p1_stats(self):
        p1_values=[]
        for _ in range(10):
            batch=self.sample()
            with torch.no_grad():
                p1=self.real_p1(batch)
            p1_values+=list(p1)
        values=np.array(p1_values)
        return values.mean(),values.std()

    def normalize_p1(self,p1):
        return (p1-self.p1_mean)/self.p1_std

    def save(self,stats):
        #TODO split this into base saving actions and MNIST/DP specific saving stuff
        size = int(math.sqrt(self.output_dim))
        fake_2 = stats['fake_data'].view(self.batch_size, -1, size, size)
        fake_2 = fake_2.int()
        fake_2 = fake_2.cpu().detach().clone()
        fake_2 = torchvision.utils.make_grid(fake_2, nrow=8, padding=2)
        self.writer.add_image('G/images', fake_2, stats['iteration'])

        dev_proj_err, dev_disc_cost=self.validation()
        #Generating images for tensorboard display
        mean,std=self.p1_mean,self.p1_std
        lv=torch.tensor([mean-std,mean,mean+std,mean+2*std]).view(-1,1).float().to(self.device)
        with torch.no_grad():
            noisev=self.fixed_noise
            lv_v=lv
        noisev=noisev.float()
        gen_images=self.G(noisev,lv_v).view((4,-1,size,size))
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


    @abstractmethod
    def proj_loss(self,fake_data,real_p1):
        pass

    @abstractmethod
    def real_p1(self,images):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def norm_data(self,data):
        pass

