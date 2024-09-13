import os,sys
import numpy as np
import time

import torch
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader

import T1_dataloader as dloader
import T2_ResNet     as mynet


print('torch-version: ', torch.__version__)
print('torch.backends.mps is_available: ', torch.backends.mps.is_available())

version = 1
main_path  = './'
BATCH_SIZE = 10


print('>>> 1. load net')
net_path = os.path.join(main_path, 'save_net')
net_file = ' '

if os.path.exists(net_path):
    pass
else:
    os.mkdir(net_path)

net = mynet.CNN_cosmo()

if os.path.exists(net_path+net_file):
    print('\t load the existed net_params to net')
    print('\t netparam_file: \n\t\t', os.path.join(net_path, net_file))
    net.load_state_dict(torch.load(os.path.join(net_path, net_file)))
else:
    print('\t No net_params exists, use the initial net')

if torch.backends.mps.is_available(): 
    device = 'mps'  # for apple M-chips
else:
    device = 'cpu'  
    
net = net.to(device)

print('>>> 2. load data_set')
train_set = dloader.DatasetLoader(set_type='train', sim_type='SIMBA')
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, drop_last=True, 
                          pin_memory=True, num_workers=0, shuffle=True)

test_set = dloader.DatasetLoader(set_type='test',  sim_type='SIMBA')
test_loader = DataLoader(dataset=test_set,  batch_size=BATCH_SIZE, drop_last=True, 
                          pin_memory=True, num_workers=0)


print('>>> 3. choose optimizer and loss-function')
#optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

loss_func = torch.nn.MSELoss()



print('>>> 4. train net')
loss_mean = [] # [epoch, train_lossmean, test_lossmean]
for epoch in range(0, 100, 1):
    print('\t '+'='*20+' Epoch=', epoch, '='*20)
    
    net = net.train()

    train_loss = []
    for step, (batch_data, batch_targ) in enumerate(train_loader):
        time_ = time.strftime('%H:%M:%S', time.localtime(time.time()))
        
        batch_data = batch_data.type(torch.FloatTensor).to(device)
        batch_targ = batch_targ.type(torch.FloatTensor).to(device)
        
        out  = net(batch_data)
        loss = loss_func(out, batch_targ)
        
        if ((step+1)%10==0):
            print('\t step-'+str(step)+', time-'+time_+'\n\t\t loss={:.4f}'.format(loss.data))
            
        if device != 'cpu':
            train_loss.append(loss.cpu().data)
        else:
            train_loss.append(loss.data)
        
        ''' >>>  clean the old gradients  <<< '''
        optimizer.zero_grad()
        ''' >>>     back -propagation     <<< '''
        loss.backward()
        ''' >>> take new gradients effect <<< '''
        optimizer.step()
        
        
    if ((epoch+1)%10==0):
        
        net = net.eval()
        
        with torch.no_grad():
            test_loss = []
            for step, (batch_data, batch_targ) in enumerate(test_loader):
                batch_data = batch_data.type(torch.FloatTensor).to(device)
                batch_targ = batch_targ.type(torch.FloatTensor).to(device)
                out  = net(batch_data)
                loss = loss_func(out, batch_targ)
                if device != 'cpu':
                    test_loss.append(loss.cpu().data)
                else:
                    test_loss.append(loss.data)
            test_loss_mean  = np.mean(test_loss)
            
            
        train_loss_mean = np.mean(train_loss)
        loss_mean.append([epoch, train_loss_mean, test_loss_mean])

        print('epoch={:>03d}'.format(epoch)+':  save net params')
        net_savename = 'CNN_cosmo-V{:>02d}'.format(version)+'_epoch-{:>03d}'.format(epoch+1)+'_net_params.pkl'
        mynet.save_net_params(net, os.path.join(net_path,net_savename))

print('>>> 5. save the trained net and loss-evolution')
loss_mean = np.array(loss_mean)
np.savetxt(os.path.join(net_path,'CNN_cosmo-V{:>02d}'.format(version)+'_epoch-{:>03d}'.format(epoch+1)+'_TrainLossMean_TestLossMean.txt'), loss_mean)
print('\t Training finished !!!')


    
