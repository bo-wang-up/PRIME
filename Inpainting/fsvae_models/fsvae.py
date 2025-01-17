
import torch
import torch.nn as nn
from .snn_layers import *
from .fsvae_prior import *
from .fsvae_posterior import *
import torch.nn.functional as F

import global_v as glv

class FSVAE(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = glv.network_config['in_channels']
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']

        # hidden_dims = [32, 32, 32]
        hidden_dims = [32, 32, 32] # small_network_1
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        is_first_conv = True
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike(),
                        is_first_conv=is_first_conv)
            )
            in_channels = h_dim
            is_first_conv = False
        
        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1]*16,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        self.prior = PriorBernoulliSTBP(self.k)
        
        self.posterior = PosteriorBernoulliSTBP(self.k)

        # Build Decoder
        modules = []
        
        self.decoder_input = tdLinear(latent_dim, 
                                        hidden_dims[-1] * 16, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * 16),
                                        spike=LIFSpike())
        
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                    tdConvTranspose(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(hidden_dims[i+1]),
                                    spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            tdConvTranspose(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1,
                                            bias=True,
                                            bn=tdBatchNorm(hidden_dims[-1]),
                                            spike=LIFSpike()),
                            tdConvTranspose(hidden_dims[-1], 
                                            out_channels=glv.network_config['in_channels'],
                                            kernel_size=3, 
                                            padding=1,
                                            bias=True,
                                            bn=None,
                                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()

    def forward(self, x, scheduled=False):
        sampled_z, q_z, p_z = self.encode(x, scheduled)
        x_recon = self.decode(sampled_z)
        return x_recon, q_z, p_z, sampled_z
    
    def encode(self, x, scheduled=False):
        x = self.encoder(x) # (N,C,H,W,T)
        x = torch.flatten(x, start_dim=1, end_dim=3) # (N,C*H*W,T)
        latent_x = self.before_latent_layer(x) # (N,latent_dim,T)
        sampled_z, q_z = self.posterior(latent_x) # sampled_z:(B,C,1,1,T), q_z:(B,C,k,T)

        p_z = self.prior(sampled_z, scheduled, self.p)
        return sampled_z, q_z, p_z

    def decode(self, z):
        result = self.decoder_input(z) # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], 4, 4, self.n_steps) # (N,C,H,W,T)
        result = self.decoder(result)# (N,C,H,W,T)
        result = self.final_layer(result)# (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result))        
        return out

    def sample(self, batch_size=64):
        sampled_z = self.prior.sample(batch_size)
        sampled_x = self.decode(sampled_z)
        return sampled_x, sampled_z
        
    def loss_function_mmd(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        q_z_ber = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        p_z_ber = torch.mean(p_z, dim=2) # (N, latent_dim, T)

        #kld_loss = torch.mean((q_z_ber - p_z_ber)**2)
        mmd_loss = torch.mean((self.psp(q_z_ber)-self.psp(p_z_ber))**2)
        loss = recons_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Distance_Loss': mmd_loss}

    def loss_function_kld(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        prob_q = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        prob_p = torch.mean(p_z, dim=2) # (N, latent_dim, T)
        
        kld_loss = prob_q * torch.log((prob_q+1e-2)/(prob_p+1e-2)) + (1-prob_q)*torch.log((1-prob_q+1e-2)/(1-prob_p+1e-2))
        kld_loss = torch.mean(torch.sum(kld_loss, dim=(1,2)))

        loss = recons_loss + 1e-4 * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Distance_Loss': kld_loss}
    def weight_clipper(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4,4)

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p-init_p) * epoch / max_epoch + init_p
        
class FSVAE_EE(FSVAE):
    def __init__(self):
        super().__init__()

    def decode(self, z):
        n_steps = z.shape[-1]
        result = self.decoder_input(z) # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], 4, 4, n_steps) # (N,C,H,W,T)
        result = self.decoder(result)# (N,C,H,W,T)
        result = self.final_layer(result)# (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result))        
        return out

    def sample(self, batch_size=64):
        sampled_z = self.prior.sample(batch_size)
        timesteps = sampled_z.shape[-1]
        recon_list = [torch.zeros(1).to(sampled_z.device)]
        eet = 0
        for t in range(timesteps):
            sampled_x = self.decode(sampled_z[...,:t+1])
            recon_list.append(sampled_x)
            eet = t+1
            if t>1:
                m_norm_0 = self.compare_images(recon_list[t-2], recon_list[t-1])
                m_norm_1= self.compare_images(recon_list[t-1], recon_list[t])
                m_norm_2= self.compare_images(recon_list[t], recon_list[t+1])
                if (m_norm_2 < m_norm_1) and (m_norm_1 < m_norm_0) and (m_norm_2 < 0.1):
                    break
                else:
                    continue
        
        
        return sampled_x, sampled_z
    
    def compare_images(self, img1, img2):
        # calculate the difference and its norms
        diff = img1 - img2  
        m_norm = torch.sum(torch.abs(diff)) / 100

        return m_norm       


class Pruning_FSVAE(FSVAE):
    def __init__(self):
        super().__init__()
        in_channels = glv.network_config['in_channels']
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']

        hidden_dims = [32, 32, 32] 

        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                Pruning_tdConv(in_channels,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = Pruning_tdLinear(hidden_dims[-1]*16,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        
        self.prior = PriorBernoulliSTBP(self.k)
        
        self.posterior = PosteriorBernoulliSTBP(self.k)
        
        # Build Decoder
        modules = []
        
        self.decoder_input = Pruning_tdLinear(latent_dim, 
                                        hidden_dims[-1] * 16, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * 16),
                                        spike=LIFSpike())

        
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                    Pruning_tdConvTranspose(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(hidden_dims[i+1]),
                                    spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            Pruning_tdConvTranspose(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1,
                                            bias=True,
                                            bn=tdBatchNorm(hidden_dims[-1]),
                                            spike=LIFSpike()),
                            Pruning_tdConvTranspose(hidden_dims[-1], 
                                            out_channels=glv.network_config['in_channels'],
                                            kernel_size=3, 
                                            padding=1,
                                            bias=True,
                                            bn=None,
                                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()

    def decode(self, z):
        result = self.decoder_input(z) # (N,C*H*W,T)
        n_steps = z.shape[-1]
        result = result.view(result.shape[0], self.hidden_dims[-1], 4, 4, n_steps) # (N,C,H,W,T)
        result = self.decoder(result)# (N,C,H,W,T)
        result = self.final_layer(result)# (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result))        
        return out

###############
# FSVAE_fashion is to validate the model on fashion mnist dataset with weight tuning
##############
class FSVAE_fashion(FSVAE):
    def __init__(self, noise=0):
        super().__init__()
        in_channels = glv.network_config['in_channels']
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']


        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1]*4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        self.prior = PriorBernoulliSTBP(self.k)
        
        self.posterior = PosteriorBernoulliSTBP(self.k)
        
        # Build Decoder
        modules = []
        
        self.decoder_input = tdLinear(latent_dim, 
                                        hidden_dims[-1] * 4, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * 4),
                                        spike=LIFSpike())
        
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                    tdConvTranspose(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(hidden_dims[i+1]),
                                    spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            tdConvTranspose(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1,
                                            bias=True,
                                            bn=tdBatchNorm(hidden_dims[-1]),
                                            spike=LIFSpike()),
                            tdConvTranspose(hidden_dims[-1], 
                                            out_channels=glv.network_config['in_channels'],
                                            kernel_size=3, 
                                            padding=1,
                                            bias=True,
                                            bn=None,
                                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()

    def decode(self, z):
        result = self.decoder_input(z) # (N,C*H*W,T)
        n_steps = z.shape[-1]
        result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, n_steps) # (N,C,H,W,T)
        result = self.decoder(result)# (N,C,H,W,T)
        result = self.final_layer(result)# (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result))        
        return out

class FSVAE_fashion_EE(FSVAE_fashion):
    def __init__(self):
        super().__init__()

    def forward(self, x, scheduled=False):

        timesteps = x.shape[4]
        recon_list = [torch.zeros(1).to(x.device)]

        eet = 0
        for t in range(timesteps):
            sampled_z, q_z, p_z = self.encode(x[...,:t+1], scheduled)
            x_recon = self.decode(sampled_z)
            recon_list.append(x_recon)
            eet = t+1

            if t>1:
                m_norm_0 = self.compare_images(recon_list[t-2], recon_list[t-1])
                m_norm_1= self.compare_images(recon_list[t-1], recon_list[t])
                m_norm_2= self.compare_images(recon_list[t], recon_list[t+1])
                if (m_norm_2 < m_norm_1) and (m_norm_1 < m_norm_0) and (m_norm_2 < 0.2):
                    break
                else:
                    continue


        return x_recon, q_z, p_z, sampled_z, eet, m_norm_2

    def sample(self, batch_size=64):
        sampled_z = self.prior.sample(batch_size)
        timesteps = sampled_z.shape[-1]
        recon_list = [torch.zeros(1).to(sampled_z.device)]
        eet = 0
        for t in range(timesteps):
            sampled_x = self.decode(sampled_z[...,:t+1])
            recon_list.append(sampled_x)
            eet = t+1
            if t>1:
                m_norm_0 = self.compare_images(recon_list[t-2], recon_list[t-1])
                m_norm_1= self.compare_images(recon_list[t-1], recon_list[t])
                m_norm_2= self.compare_images(recon_list[t], recon_list[t+1])
                if (m_norm_2 < m_norm_1) and (m_norm_1 < m_norm_0) and (m_norm_2 < 1.0):
                    break
                else:
                    continue
        
        
        return sampled_x, sampled_z
    
    def compare_images(self, img1, img2):
        # calculate the difference and its norms
        diff = img1 - img2  
        m_norm = torch.sum(torch.abs(diff)) / 100

        return m_norm

###############
#Pruning_****_large aims to simulate the rram weights
#Put the model to large dataset
# prove the scalibility of model
##############
class Pruning_FSVAE_large(FSVAE):
    def __init__(self, noise=0):
        super().__init__()
        in_channels = glv.network_config['in_channels']
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']


        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                Pruning_tdConv_large(in_channels,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike(),
                        noise=noise)
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = Pruning_tdLinear_large(hidden_dims[-1]*4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike(),
                                            noise=noise)

        self.prior = PriorBernoulliSTBP(self.k)
        
        self.posterior = PosteriorBernoulliSTBP(self.k)
        
        # Build Decoder
        modules = []
        
        self.decoder_input = Pruning_tdLinear_large(latent_dim, 
                                        hidden_dims[-1] * 4, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * 4),
                                        spike=LIFSpike(),
                                        noise=noise)
        
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                    Pruning_tdConvTranspose_large(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(hidden_dims[i+1]),
                                    spike=LIFSpike(),
                                    noise=noise)
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            Pruning_tdConvTranspose_large(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1,
                                            bias=True,
                                            bn=tdBatchNorm(hidden_dims[-1]),
                                            spike=LIFSpike(),
                                            noise=noise),
                            Pruning_tdConvTranspose_large(hidden_dims[-1], 
                                            out_channels=glv.network_config['in_channels'],
                                            kernel_size=3, 
                                            padding=1,
                                            bias=True,
                                            bn=None,
                                            spike=None,
                                            noise=noise)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()

    def decode(self, z):
        result = self.decoder_input(z) # (N,C*H*W,T)
        n_steps = z.shape[-1]
        result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, n_steps) # (N,C,H,W,T)
        result = self.decoder(result)# (N,C,H,W,T)
        result = self.final_layer(result)# (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result))        
        return out

class Pruning_FSVAE_large_EE(Pruning_FSVAE_large):
    def __init__(self, noise=0):
        super().__init__(noise=noise)

    def forward(self, x, scheduled=False):

        timesteps = x.shape[4]
        recon_list = [torch.zeros(1).to(x.device)]

        eet = 0
        for t in range(timesteps):
            sampled_z, q_z, p_z = self.encode(x[...,:t+1], scheduled)
            x_recon = self.decode(sampled_z)
            recon_list.append(x_recon)
            eet = t+1

            if t>1:
                m_norm_0 = self.compare_images(recon_list[t-2], recon_list[t-1])
                m_norm_1= self.compare_images(recon_list[t-1], recon_list[t])
                m_norm_2= self.compare_images(recon_list[t], recon_list[t+1])
                if (m_norm_2 < m_norm_1) and (m_norm_1 < m_norm_0) and (m_norm_2 < 1.0):
                    break
                else:
                    continue


        return x_recon, q_z, p_z, sampled_z, eet, m_norm_2

    def sample(self, batch_size=64):
        sampled_z = self.prior.sample(batch_size)
        timesteps = sampled_z.shape[-1]
        recon_list = [torch.zeros(1).to(sampled_z.device)]
        eet = 0
        for t in range(timesteps):
            sampled_x = self.decode(sampled_z[...,:t+1])
            recon_list.append(sampled_x)
            eet = t+1
            if t>1:
                m_norm_0 = self.compare_images(recon_list[t-2], recon_list[t-1])
                m_norm_1= self.compare_images(recon_list[t-1], recon_list[t])
                m_norm_2= self.compare_images(recon_list[t], recon_list[t+1])
                if (m_norm_2 < m_norm_1) and (m_norm_1 < m_norm_0) and (m_norm_2 < 0.1):
                    break
                else:
                    continue
        
        
        return sampled_x, sampled_z
    
    def compare_images(self, img1, img2):
        # calculate the difference and its norms
        diff = img1 - img2  
        m_norm = torch.sum(torch.abs(diff)) / 100

        return m_norm




class Pruning_FSVAE_EE(Pruning_FSVAE):
    def __init__(self):
        super().__init__()

    
    def forward(self, x, scheduled=False):

        timesteps = x.shape[4]
        recon_list = [torch.zeros(1).to(x.device)]

        eet = 0
        for t in range(timesteps):
            sampled_z, q_z, p_z = self.encode(x[...,:t+1], scheduled)
            x_recon = self.decode(sampled_z)
            recon_list.append(x_recon)
            eet = t+1

            if t>1:
                m_norm_0 = self.compare_images(recon_list[t-2], recon_list[t-1])
                m_norm_1= self.compare_images(recon_list[t-1], recon_list[t])
                m_norm_2= self.compare_images(recon_list[t], recon_list[t+1])
                if (m_norm_2 < m_norm_1) and (m_norm_1 < m_norm_0) and (m_norm_2 < 0.2):
                    break
                else:
                    continue


        return x_recon, q_z, p_z, sampled_z, eet, m_norm_2

    def sample(self, batch_size=64):
        sampled_z = self.prior.sample(batch_size)
        timesteps = sampled_z.shape[-1]
        recon_list = [torch.zeros(1).to(sampled_z.device)]
        eet = 0
        for t in range(timesteps):
            sampled_x = self.decode(sampled_z[...,:t+1])
            recon_list.append(sampled_x)
            eet = t+1
            if t>1:
                m_norm_0 = self.compare_images(recon_list[t-2], recon_list[t-1])
                m_norm_1= self.compare_images(recon_list[t-1], recon_list[t])
                m_norm_2= self.compare_images(recon_list[t], recon_list[t+1])
                if (m_norm_2 < m_norm_1) and (m_norm_1 < m_norm_0) and (m_norm_2 < 0.3):
                    break
                else:
                    continue
        
        
        return sampled_x, sampled_z
    
    def compare_images(self, img1, img2):
        # calculate the difference and its norms
        diff = img1 - img2  
        m_norm = torch.sum(torch.abs(diff)) / 100

        return m_norm
    
