import torch
import torch.nn as nn

cVAE_hyper_params = {
    'batch_size': 128, 
    'learning_rate': 1e-4, 
    'max_epoch': 150, 
    'channel': 1,
    'latent_dim': 20, 
    'label_dim': 5,
    'label_up_dim': 50
}

Implemented_nets = {
    'cVAE': cVAE_hyper_params
}

def save_net_params(net, path):
    torch.save(net.state_dict(), path)
    print('Net_params has been saved in '+path+' .')
    return

class ConditionalConvVAE(nn.Module):
    def __init__(self, hyper_param_dict):
        super(ConditionalConvVAE, self).__init__()
        
        # Load hyperparameters
        image_channels = hyper_param_dict['channel']
        latent_dim = hyper_param_dict['latent_dim']
        label_dim = hyper_param_dict['label_dim']
        label_up_dim = hyper_param_dict['label_up_dim']

        # Embedding layer for labels
        self.label_embed = nn.Linear(label_dim, label_up_dim)  # Project label to a higher dimension
        
        # Encoder: Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels + label_up_dim, 32, kernel_size=4, stride=2, padding=1),  # [batch, 32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),                   # [batch, 64, 7, 7]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),                  # [batch, 128, 3, 3]
            nn.ReLU(),
            nn.Flatten(),                                                            # [batch, 1152]
        )
        
        # Latent space
        self.fc_mu = nn.Linear(128 * 3 * 3 + label_up_dim, latent_dim)        # Mean of latent space
        self.fc_logvar = nn.Linear(128 * 3 * 3 + label_up_dim, latent_dim)    # Log variance of latent space
        
        # Decoder: Fully connected layer followed by transposed convolutions
        self.fc_decode = nn.Linear(latent_dim + label_up_dim, 128 * 7 * 7)    # Start from a bigger dimension for transposed convs
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # [batch, 64, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # [batch, 32, 28, 28]
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=1, padding=1),  # [batch, 1, 28, 28]
            nn.Sigmoid()  # Output pixel values in [0, 1]
        )
    
    def encode(self, x, labels):
        labels_embedded = self.label_embed(labels)
        
        # Expand labels to match input dimensions and concatenate along the channel dimension
        labels_expanded = labels_embedded.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
        # print(labels_expanded.shape)
        x = torch.cat([x, labels_expanded], dim=1)  # Concatenate labels along the channel dimension
        # print(x.shape)
        
        h = self.encoder(x)
        h = torch.cat([h, labels_embedded], dim=1)  # Concatenate flattened conv output and labels
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Random noise
        z = mu + eps * std  # Reparameterization trick
        return z
    
    def decode(self, z, labels):
        labels_embedded = self.label_embed(labels)
        z = torch.cat([z, labels_embedded], dim=1)  # Concatenate z and embedded labels
        
        h = self.fc_decode(z)
        h = h.view(-1, 128, 7, 7)  # Reshape to fit into the transposed convolutions
        x_recon = self.decoder(h)
        return x_recon
    
    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, labels)
        return x_recon, mu, logvar, z