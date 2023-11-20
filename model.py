"""beta-VAE model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE(nn.Module):
    """Model proposed in original VAE paper(Higgins et al, ICLR, 2017)."""

    """
    class BetaVAE(nn.Module):
        def __init__(self, z_dim=10, nc=1):
            super(BetaVAE, self).__init__()
            self.z_dim = z_dim
            self.nc = nc
            # Encoder
            self.enc_conv1 = nn.Conv2d(nc, 8, 3, 2, 1)               # B, 8, 14, 14
            self.enc_conv2 = nn.Conv2d(8, 16, 3, 2, 1)               # B, 16, 7, 7
            self.enc_conv3 = nn.Conv2d(16, 32, 3, 2)                 # B, 32, 3, 3
            self.enc_fc1 = nn.Linear(32*3*3, 128)                    # B, 128
            self.enc_mean = nn.Linear(128, z_dim)                    # B, z_dim
            self.enc_std = nn.Linear(128, z_dim)                     # B, z_dim
            # Decoder
            self.dec_fc1 = nn.Linear(z_dim, 128)                     # B, 128
            self.dec_fc2 = nn.Linear(128, 32*3*3)                    # B, 32*3*3
            self.dec_convt1 = nn.ConvTranspose2d(32, 16, 3, 2)       # B, 16, 7, 7
            self.dec_convt2 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)  # B, 8, 14, 14
            self.dec_convt3 = nn.ConvTranspose2d(8, nc, 3, 2, 1, 1)  # B, 1, 28, 28
            # BN
            self.bn = nn.BatchNorm2d(16)
    
    """

    def __init__(self, out_dim=10, z_dim=15, nc=1):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc

        # Encoder
        self.enc_conv1 = nn.Conv2d(nc, 8, 3, 2, 1)  # B, 8, 14, 14
        self.enc_conv2 = nn.Conv2d(8, 16, 3, 2, 1)  # B, 16, 7, 7
        self.enc_conv3 = nn.Conv2d(16, 32, 3, 2)  # B, 32, 3, 3
        self.enc_fc1 = nn.Linear(32 * 3 * 3, 128)  # B, 128
        self.enc_mean = nn.Linear(128, z_dim)  # B, z_dim
        self.enc_std = nn.Linear(128, z_dim)  # B, z_dim
        # Decoder
        self.dec_fc1 = nn.Linear(z_dim, 128)  # B, 128
        self.dec_fc2 = nn.Linear(128, 32 * 3 * 3)  # B, 32*3*3
        self.dec_convt1 = nn.ConvTranspose2d(32, 16, 3, 2)  # B, 16, 7, 7
        self.dec_convt2 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)  # B, 8, 14, 14
        self.dec_convt3 = nn.ConvTranspose2d(8, nc, 3, 2, 1, 1)  # B, 1, 28, 28
        # BN
        self.bn = nn.BatchNorm2d(16)

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )



        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),

            ## Step 6: Fill out the Transpose Conv. Layer
            nn.ConvTranspose2d( 32, nc, 4, 2, 1 ),        # in channels = 32 , out_channels = 1, kernel_size =4, stride = 2, padding = 1
        )

        # self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def _encoder(self, x):
        x = F.relu(self.enc_conv1(x.view(-1, self.nc, 28, 28)))
        x = F.relu(self.enc_conv2(x))
        x = self.bn(x)
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_fc1(x.view(-1, 32 * 3 * 3)))
        mean = self.enc_mean(x)
        std = F.softplus(self.enc_std(x))
        return mean, std

    def _decoder(self, z):
        x = F.relu(self.dec_fc1(z))
        x = F.relu(self.dec_fc2(x))
        x = F.relu(self.dec_convt1(x.view(-1, 32, 3, 3)))
        x = self.bn(x)
        x = F.relu(self.dec_convt2(x))
        x = self.dec_convt3(x)
        # Using sigmoid function so that the output should be in [0,1]
        x = torch.sigmoid(x.view(-1, self.nc * 28 * 28))
        return x

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        
        ## reprameterize trick
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x):
        return self._encoder(x)

    def _decode(self, z):
        return self._decoder(z)


def kaiming_init(m):
    ## nn.init.kaiming_normal_
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
        

# if __name__ == '__main__':
#     pass
#     