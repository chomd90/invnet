import math
import torchvision
from models.wgan import *
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
import os
from utils import calc_gradient_penalty,gen_rand_noise,\
                        weights_init,generate_image
import time
from abc import ABC,abstractmethod


class BaseInvNet(ABC):

    def __init__(self, batch_size, output_path, data_dir, lr, critic_iters, proj_iters, output_dim, hidden_size, device, lambda_gp, restore_mode=False,hparams={}):
        self.writer = SummaryWriter()
        print('output dir:', self.writer.logdir)
        #TODO Expand hyperparameter tracking
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
            self.G = GoodGenerator(hidden_size, self.output_dim, ctrl_dim=1).to(device)
            self.D = GoodDiscriminator(hidden_size).to(device)
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        self.optim_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0, 0.9))
        self.optim_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0, 0.9))
        self.optim_pj = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0, 0.9))

        self.fixed_noise = gen_rand_noise(4)

        self.start = timer()

    def train(self, iters):

        for iteration in range(iters):
            print('iteration:', iteration)

            gen_cost, real_p1 = self.generator_update()
            start_time = time.time()
            proj_cost = self.proj_update()
            proj_time = time.time() - start_time
            stats = self.critic_update()
            add_stats = {'start': start_time,
                         'iteration': iteration,
                         'gen_cost': gen_cost,
                         'proj_cost': proj_cost}
            stats.update(add_stats)

            self.log(stats)
            if iteration % 100 == 0:
                val_proj_err=self.save(stats)




    def generator_update(self):
        start=timer()
        for p in self.D.parameters():
            p.requires_grad_(False)

        real_data= self.sample()
        real_images=real_data[0].to(self.device)
        real_lengths=self.real_p1(real_images)
        real_p1=real_lengths.to(self.device)
        mone = torch.FloatTensor([1]) * -1
        mone=mone.to(self.device)

        for i in range(1):
            self.G.zero_grad()
            noise = gen_rand_noise(self.batch_size).to(self.device)
            noise.requires_grad_(True)
            fake_data = self.G(noise, real_p1)
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
            real_data = self.sample()
            # gen fake data and load real data
            noise = gen_rand_noise(self.batch_size).to(self.device)

            real_images = real_data[0].to(self.device)
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
            gradient_penalty = calc_gradient_penalty(self.D, real_images, fake_data, self.batch_size, self.lambda_gp)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward()
            w_dist = disc_fake - disc_real

            self.optim_d.step()
        end = timer()
        # print('---train D elapsed time:', end - start)
        stats={'w_dist': w_dist,
               'disc_cost':disc_cost,
               'fake_data':fake_data,
               'real_data':real_data,
               'disc_real':disc_real,
               'disc_fake':disc_fake,
               'gradient_penalty':gradient_penalty}
        return stats

    def proj_update(self):
        start=timer()
        real_data = self.sample()
        with torch.no_grad():
            images = real_data[0].to(self.device)
            real_lengths = self.real_p1(images).view(-1, 1)
        for iteration in range(self.proj_iters):
            self.G.zero_grad()
            noise=gen_rand_noise(self.batch_size).to(self.device)
            noise.requires_grad=True
            fake_data = self.G(noise, real_lengths)
            normed_fake= self.norm_data(fake_data)
            pj_loss=self.proj_loss(normed_fake,real_lengths)
            pj_loss.backward()
            self.optim_pj.step()

        end=timer()
        # print('--projection update elapsed time:',end-start)
        return pj_loss

    def save(self, stats):
        #TODO split this into base saving actions and MNIST/DP specific saving stuff
        size = int(math.sqrt(self.output_dim))
        fake_2 = torch.argmax(stats['fake_data'].view(self.batch_size, 1, size, size), dim=1).unsqueeze(1)
        fake_2 = fake_2.int()
        fake_2 = fake_2.cpu().detach().clone()
        fake_2 = torchvision.utils.make_grid(fake_2, nrow=8, padding=2)
        self.writer.add_image('G/images', fake_2, stats['iteration'])

        dev_disc_costs = []
        for _, images in enumerate(self.val_iter):
            imgs = torch.Tensor(images[0])
            imgs = imgs.to(self.device)
            with torch.no_grad():
                imgs_v = imgs

            D = self.D(imgs_v)
            _dev_disc_cost = -D.mean().cpu().data.numpy()
            dev_disc_costs.append(_dev_disc_cost)
        gen_images = generate_image(self.G, 4, noise=self.fixed_noise, device=self.device)
        real_images = stats['real_data'][0]
        mean = gen_images.mean()
        std = gen_images.std()
        gen_images = (gen_images - mean) / (std / 0.7)

        real_grid_images = torchvision.utils.make_grid(real_images[:4], nrow=8, padding=2)
        fake_grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
        real_grid_images = real_grid_images.long()
        fake_grid_images = fake_grid_images.long()
        self.writer.add_image('real images', real_grid_images, stats['iteration'])
        self.writer.add_image('fake images', fake_grid_images, stats['iteration'])
        torch.save(self.G, self.output_path + 'generator.pt')
        torch.save(self.D, self.output_path + 'discriminator.pt')

        val_batch = self.sample(train=False)
        images = val_batch[0].detach().to(self.device)
        real_lengths = self.real_p1(images)

        noise = gen_rand_noise(real_lengths.shape[0]).to(self.device)
        fake_data = self.G(noise, real_lengths.to(self.device))
        fake_avg = fake_data.mean()
        fake_std = fake_data.std()

        normed_fake = (fake_data - fake_avg) / (fake_std / 0.8)
        normed_fake = normed_fake.view(-1, 32, 32).to(self.device)
        fake_lengths = self.dp_layer(normed_fake)
        diff = fake_lengths - real_lengths.squeeze()
        val_proj_err = (diff ** 2).mean()

        metric_dict = {'generator_cost': stats['gen_cost'],
                       'discriminator_cost': stats['disc_cost'], 'validation_projection_error': val_proj_err}
        self.writer.add_hparams(self.hparams, metric_dict,global_step=stats['iteration'])
        return val_proj_err

    def log(self,stats):
        # ------------------VISUALIZATION----------
        self.writer.add_scalar('data/gen_cost', stats['gen_cost'], stats['iteration'])
        self.writer.add_scalar('data/disc_cost', stats['disc_cost'], stats['iteration'])
        self.writer.add_scalar('data/disc_fake', stats['disc_fake'], stats['iteration'])
        self.writer.add_scalar('data/disc_real', stats['disc_real'], stats['iteration'])
        self.writer.add_scalar('data/gradient_pen', stats['gradient_penalty'], stats['iteration'])
        self.writer.add_scalar('data/proj_error',stats['proj_cost'],stats['iteration'])

    @abstractmethod
    def proj_loss(self,fake_data,real_p1):
        pass

    @abstractmethod
    def sample(self,train):
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

