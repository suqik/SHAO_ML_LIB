import numpy  as np
import torch
from torch.utils.data import DataLoader, random_split

class DatasetLoader(DataLoader):
    def __init__(self, image, label):
        self.image = image
        self.label = label
        
    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        data  = self.image[idx]
        lb    = self.label[idx]
        data  = torch.tensor(data[np.newaxis, ...], dtype=torch.float32)
        lb = torch.tensor(lb, dtype=torch.float32)
        
        return data, lb

def sep_train_val(dataset:DatasetLoader, batch_size, train_ratio=0.8):
    train_size = int(train_ratio*len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)

    return train_loader, val_loader
