import os
import os.path
import numpy as np

import torch
import torchvision

import global_v as glv
from datasets import load_dataset_snn
from utils import AverageMeter
import fsvae_models.fsvae as fsvae


def add_noise(inputs, noise_factor=0.2):
    noise = inputs + torch.randn_like(inputs) * noise_factor
    return noise

def impaint(inputs):
    inputs = inputs.clone()
    inputs[:, :, 12:20, 12:20] = -1 # v1
    return inputs

EE_threshold = 2.0
folder_path = f'demo_imgs/software/EE_{EE_threshold}'

def test_sample(network, test_loader):
    n_steps = glv.network_config['n_steps']
    loss_meter = AverageMeter()
    recons_meter = AverageMeter()
    dist_meter = AverageMeter()
    diff_meter = AverageMeter()
    timestep_meter = AverageMeter()
    
    avg_diff = 0
    avg_timestep = 0
    sample_timestep = 0

    network = network.eval()
    real_images = []
    noise_images = []
    recon_images = []
    with torch.no_grad():
        for batch_idx, (real_img, labels) in enumerate(test_loader):
            real_img = real_img.to(init_device, non_blocking=True)
            noise_img = impaint(real_img)
            noise_img = noise_img.to(init_device, non_blocking=True)
            labels = labels.to(init_device, non_blocking=True)
            # direct spike input
            spike_input = noise_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps) # (N,C,H,W,T)

            x_recon, q_z, p_z, sampled_z,t, diff = network(spike_input, scheduled=False)



            diff_meter.update(diff.detach().cpu().item())
            timestep_meter.update(t)

            losses = network.loss_function_mmd(real_img, x_recon, q_z, p_z)
            loss_meter.update(losses['loss'].detach().cpu().item())
            recons_meter.update(losses['Reconstruction_Loss'].detach().cpu().item())
            dist_meter.update(losses['Distance_Loss'].detach().cpu().item())
            
            if (batch_idx) % 10 == 0:
                real_images.append(real_img)
                noise_images.append(noise_img)
                recon_images.append(x_recon)
                sample_timestep += t


                save_real_images = torch.cat(real_images, dim=0).contiguous()
                save_noise_images = torch.cat(noise_images, dim=0).contiguous()
                save_recon_images = torch.cat(recon_images, dim=0).contiguous()
                if not os.path.exists(folder_path):  # Check if the folder path doesn't exist
                    os.makedirs(folder_path)  # Create the folder
                torchvision.utils.save_image((save_real_images+1)/2, folder_path + '/EE_input.png', nrow=5)
                torchvision.utils.save_image((save_noise_images+1)/2, folder_path + '/EE_noise.png', nrow=5)
                torchvision.utils.save_image((save_recon_images+1)/2, folder_path + '/EE_output.png', nrow=5)
            
            print('test_sample_difference:', diff_meter.avg)
            print('test_sample_timestep:', timestep_meter.avg)
            print('test_figure_timestep:',sample_timestep)
            print(f'Test_Sample[{t}/{n_steps}] [{batch_idx+1}/{len(test_loader)}] Loss: {loss_meter.avg}, RECONS: {recons_meter.avg}, DISTANCE: {dist_meter.avg}')
        txt = f'Test_Sample[{t}/{n_steps}] [{batch_idx+1}/{len(test_loader)}] Loss: {loss_meter.avg}, RECONS: {recons_meter.avg}, DISTANCE: {dist_meter.avg}, TIMESTEP: {timestep_meter.avg}'
        with open(folder_path + '/output.txt', "w") as file:
            file.write(txt)
def sample(network):
    network = network.eval()
    real_images = []
    recon_images = []
    avg_t = 0
    with torch.no_grad():
        for i in range(128):
            sampled_x, sampled_z, t = network.sample(network_config['batch_size'])
            recon_images.append(sampled_x)
            save_recon_images = torch.cat(recon_images, dim=0).contiguous()
            torchvision.utils.save_image((save_recon_images+1)/2, f'demo_imgs/impaintv1_pruning/lightv1_ee_sample.png')
            avg_t+=t
        
        print('sample:', avg_t/128)

if __name__ == '__main__':

    init_device = torch.device("cuda:0")
    
    network_config = {"batch_size": 1, "n_steps": 16, "dataset": "MNIST",
                        "in_channels": 1, "latent_dim": 128, "input_size": 32, 
                        "k": 20, "loss_func": "mmd", "lr": 0.001}
    
    glv.init(network_config, devs=[0])

    dataset_name = glv.network_config['dataset']
    data_path = "/home/wangbo/codes/FullySpikingVAE/data" # specify the path of dataset
    
    # load MNIST dataset
    data_path = os.path.expanduser(data_path)
    _, test_loader = load_dataset_snn.load_mini_mnist(data_path, test_num=10)
        
    net = fsvae.Pruning_FSVAE_EE()
    net = net.to(init_device)
    
    # checkpoint = torch.load('checkpoint/lnv1_denoise_0.3/best.pth', map_location='cuda:1')
    checkpoint = torch.load('/home/wangbo/codes/FullySpikingVAE/checkpoint/software_pruning_impainting_norm_like_rram_scale0.003/best.pth')
    net.load_state_dict(checkpoint)    

    test_sample(net, test_loader)

