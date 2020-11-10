from invnet import InvNet
from config import InvNetConfig
import torch
import sys
import pickle

config = InvNetConfig()

# torch.cuda.set_device(config.gpu)
cuda_available = torch.cuda.is_available()
device = torch.device(config.gpu if cuda_available else "cpu")

print('training on:', device)
invnet = InvNet(config.batch_size, config.output_path, config.data_dir,
                config.lr, config.critic_iter, config.proj_iter, 32 * 32,
                config.hidden_size, device, config.lambda_gp)

for i in range(len(invnet.dataiter)):
    real_data=invnet.sample()
    images=real_data[0]
    real_lengths = invnet.real_lengths(images).view(-1, 1).to(device)
    print(len(invnet.real_length_d))

pickle.dump(invnet.real_length_d,open('/home/km3888/invnet/length_map.pkl','wb'))