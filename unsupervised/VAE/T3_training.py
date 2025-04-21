import os
import time

import torch
from T2_nets import save_net_params

class train_driver(object):
    def __init__(self, net:torch.nn.Module, 
                 device,
                 optimizer, loss_func):
        self.net = net
        self.device = device
        self.optimizer = optimizer
        self.loss_func = loss_func

    def val_round(self, val_loader):
        self.net.eval()
        device = self.device
        
        with torch.no_grad():
            val_loss = 0
            for step, (batch_data, batch_targ) in enumerate(val_loader):
                batch_data = batch_data.type(torch.FloatTensor).to(device)
                batch_targ = batch_targ.type(torch.FloatTensor).to(device)
                out, mu, logvar, _  = self.net(batch_data, batch_targ)
                loss = self.loss_func(out, batch_data, mu, logvar)
                val_loss += loss.item()

        return val_loss/len(val_loader)
    
    def run(self, train_loader, val_loader, max_epoch, 
            netname, version, net_path):
        loss_mean = []
        device = self.device

        for epoch in range(0, max_epoch):
            print('\t '+'='*20+' Epoch=', epoch, '='*20)

            ######## TRAINING ########
            self.net.train()
            train_loss = 0
            for step, (batch_data, batch_targ) in enumerate(train_loader):
                time_ = time.strftime('%H:%M:%S', time.localtime(time.time()))
                
                batch_data = batch_data.type(torch.FloatTensor).to(device)
                batch_targ = batch_targ.type(torch.FloatTensor).to(device)
                
                out, mu, logvar, _  = self.net(batch_data, batch_targ)
                loss = self.loss_func(out, batch_data, mu, logvar)
                
                train_loss += loss.item()
                
                ''' >>>  clean the old gradients  <<< '''
                self.optimizer.zero_grad()
                ''' >>>     back -propagation     <<< '''
                loss.backward()
                ''' >>> take new gradients effect <<< '''
                self.optimizer.step()

            ####### VALIDATION ########
            self.net.eval()
            with torch.no_grad():
                val_loss = 0
                for step, (batch_data, batch_targ) in enumerate(val_loader):
                    batch_data = batch_data.type(torch.FloatTensor).to(device)
                    batch_targ = batch_targ.type(torch.FloatTensor).to(device)
                    out, mu, logvar, _  = self.net(batch_data, batch_targ)
                    loss = self.loss_func(out, batch_data, mu, logvar)
                    val_loss += loss.item()

            loss_mean.append([epoch, train_loss/len(train_loader.dataset), val_loss/len(val_loader.dataset)])
            if ((epoch+1)%10==0):
                print('epoch={:>03d}'.format(epoch)+':  save net params')
                net_savename = '{:s}-V{:>02d}'.format(netname, version)+'_epoch-{:>03d}'.format(epoch+1)+'_net_params.pkl'
                save_net_params(self.net, os.path.join(net_path,net_savename))
        return loss_mean