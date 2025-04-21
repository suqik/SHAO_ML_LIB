import os,sys
import numpy as np
import configparser
import copy
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F 

from T1_dataloader import DatasetLoader, sep_train_val
from T2_nets import Implemented_nets, ConditionalConvVAE as mynet
from T3_training import train_driver

########### for test ###############
from tests.utils import load_single_ring_dataset

print('torch-version: ', torch.__version__)

version = 1

print('>>> 1. load net')
if len(sys.argv) != 2:
    print("Usage: python main.py conf")
    exit()

### read configuration file
conf_file = sys.argv[1]
conf = configparser.ConfigParser()
conf.read(conf_file)
netname = conf.get("General", "net_name").strip("\'").strip("\"")
if not netname in Implemented_nets:
    raise NotImplementedError(f"The net {netname} has not been implemented!")

if not netname in conf.sections():
    raise ValueError(f"Cannot find parameters of net {netname}!")

### set hyper parameters
net_hyper_params = copy.deepcopy(Implemented_nets[netname])
for key, value in net_hyper_params.items():
    if key in conf[netname]:
        ### FIXME: write general update function
        if key == 'learning_rate':
            net_hyper_params[key] = conf[netname].getfloat(key)
        else:
            net_hyper_params[key] = conf[netname].getint(key)
    else:
        print(f"Do not set up the hyper parameter {key}. Will use default value.")

BATCH_SIZE = net_hyper_params["batch_size"]
LEARNING_RATE = net_hyper_params["learning_rate"]
MAX_EPOCH = net_hyper_params["max_epoch"]

if "net_path" in conf[netname]:
    main_path = conf.get(netname, "net_path").strip("\"")
else:
    main_path  = './'
net_path = os.path.join(main_path, 'save_net')

if "net_file" in conf[netname]:
    net_file = conf.get(netname, "net_file").strip("\"")
else:
    net_file  = ' '

if os.path.exists(net_path):
    pass
else:
    os.makedirs(net_path)

### initialize network
net = mynet(net_hyper_params)

### if load well-trained net parameters
if os.path.exists(net_path+net_file):
    print('\t load the existed net_params to net')
    print('\t netparam_file: \n\t\t', os.path.join(net_path, net_file))
    net.load_state_dict(torch.load(os.path.join(net_path, net_file)))
else:
    print('\t No net_params exists, use the initial net')

### set device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'  
    
net = net.to(device)

print('>>> 2. load data_set')
###################################################################
### In this part you can put your own data loader ###
images, labels = load_single_ring_dataset('ring_generator_dataset')
###################################################################
train_ratio = 0.8
mydataset = DatasetLoader(images, labels)
print(f"Load {len(mydataset):d} samples.")

train_loader, val_loader = sep_train_val(mydataset, BATCH_SIZE, train_ratio)

print('>>> 3. choose optimizer and loss-function')

#optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (BCE)
    bce_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL Divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return bce_loss + kl_loss

print('>>> 4. train net')

driver = train_driver(net, device, optimizer, vae_loss)
loss_mean = driver.run(train_loader, val_loader, 
                       max_epoch=MAX_EPOCH, 
                       netname=netname, version=version, net_path=net_path)

print('>>> 5. save the trained net and loss-evolution')
loss_mean = np.array(loss_mean)
np.savetxt(os.path.join(net_path,'{:s}-V{:>02d}'.format(netname, version)+'_epoch-{:>03d}'.format(MAX_EPOCH+1)+'_TrainLossMean_TestLossMean.txt'), loss_mean)
print('\t Training finished !!!')

plt.plot(loss_mean[:,0], loss_mean[:,1])
plt.plot(loss_mean[:,0], loss_mean[:,2])
plt.savefig(os.path.join(net_path,'{:s}-V{:>02d}'.format(netname, version)+'_epoch-{:>03d}'.format(MAX_EPOCH+1)+'_TrainLossMean_TestLossMean.png'), format='png', dpi=400)