import os, sys
import numpy  as np
from torch.utils.data import Dataset


class DatasetLoader(Dataset):
    def __init__(self, set_type='train', sim_type='IllustrisTNG'):
        
        info_path = './CAMELS_multifield/'
        
        self.set_type = set_type
        self.sim_type = sim_type

        if set_type in ['train', 'test', 'estimate']:
            info_name = 'sim-idx_'+set_type+'_LH_anyone.txt'
        else:
            print("ERROR: set_type = 'test' or 'train' or 'estimate'. ")
            
        self.sim_index = np.loadtxt(info_path+info_name)
        self.param_tot = np.loadtxt(info_path+'params_LH_'+sim_type+'.txt')
        self.mmap1_tot  = np.load(info_path+'Maps_Mtot_'+sim_type+'_LH_z=0.00.npy').reshape(1000, 15, 256, 256)
        self.mmap1_tot  = np.log10(self.mmap1_tot)
        
        self.mmap2_tot  = np.load(info_path+'Maps_P_'+sim_type+'_LH_z=0.00.npy').reshape(1000, 15, 256, 256)
        self.mmap2_tot  = np.log10(self.mmap2_tot)
        
    def __len__(self):
        return len(self.sim_index)

    def __getitem__(self, idx):
        
        sim_idx = int(self.sim_index[idx][0])
        sub_idx = int(self.sim_index[idx][1])
        
        sim_Om  = self.param_tot[sim_idx][0]
        sim_s8  = self.param_tot[sim_idx][1]
        
        # map.shape = 128x128
        mmap  = self.mmap_tot[sim_idx, sub_idx, 64:-64, 64:-64]
        
        mmap1  = mmap1[np.newaxis, ...]
        mmap2  = mmap2[np.newaxis, ...]
        mmap = np.vstack([mmap1,mmap2])
        
        label = np.array([sim_Om, sim_s8])
        
        return mmap, label
