import os
import os.path
import numpy as np
import logging
import argparse
#import pycuda.driver as cuda

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

import global_v as glv
from network_parser import parse
from datasets import load_dataset_snn
from utils import aboutCudaDevices
from utils import AverageMeter
from utils import CountMulAddSNN
import fsvae_models.fsvae as fsvae
from fsvae_models.snn_layers import LIFSpike



max_accuracy = 0
min_loss = 1000

def transformRate(data2D, N_ts, max_is_present_for, seed = 0):
    """
        Transforms input data into spike trains encoding values using rate-coding.
        N_ts - number of timesteps to generate (length of the spike trains)
        max_is_present_for - expected number of spikes for the maximum value of 1.0
        seed - for reproducibility
    """
    np.random.seed(seed)
    data2D = data2D.cpu().numpy()
    data = []
    for trials in range(N_ts):
        # For each timestep of the spike train execute a series of Bernoulli trials to generate the spikes:
        trial = np.random.random(data2D.shape)
        data.append( (data2D * max_is_present_for / N_ts > trial).astype(dtype=np.uint8) )
    res = np.array(data, dtype=np.uint8) # (Ns, examples, data)
    res = torch.from_numpy(res).float()
    res = res.permute(1,2,3,4,0) # (examples, Ns, data)
    return res

def add_noise(inputs, noise_factor=0.2):
    noise = inputs + torch.randn_like(inputs) * noise_factor
    # noise = torch.clip(noise, 0., 1.)
    return noise

def impaint(inputs):
    inputs = inputs.clone()
    inputs[:, :, 12:20, 12:20] = -1 # v1
    return inputs

def add_hook(net):
    count_mul_add = CountMulAddSNN()
    hook_handles = []
    for m in net.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.ConvTranspose3d) or isinstance(m, LIFSpike):
            handle = m.register_forward_hook(count_mul_add)
            hook_handles.append(handle)
    return count_mul_add, hook_handles



def write_weight_hist(net, index):
    for n, m in net.named_parameters():
        root, name = os.path.splitext(n)
        writer.add_histogram(root + '/' + name, m, index)

def train(network, trainloader, opti, epoch, direct=True):
    n_steps = glv.network_config['n_steps']
    max_epoch = glv.network_config['epochs']
    
    loss_meter = AverageMeter()
    recons_meter = AverageMeter()
    dist_meter = AverageMeter()

    mean_q_z = 0
    mean_p_z = 0
    mean_sampled_z = 0

    network = network.train()
    
    for batch_idx, (real_img, labels) in enumerate(trainloader):   
        opti.zero_grad()
        noise_img = impaint(real_img)
        noise_img = noise_img.to(init_device, non_blocking=True)
        real_img = real_img.to(init_device, non_blocking=True)
        labels = labels.to(init_device, non_blocking=True)
        # direct spike input or rate_based encoding
        if direct == 1:
            direct_encoding = 1
            spike_input = noise_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps) # (N,C,H,W,T)
        else:
            direct_encoding = 0
            spike_input = transformRate(noise_img, n_steps, 0.3, seed=0).to(init_device, non_blocking=True)
        x_recon, q_z, p_z, sampled_z = network(spike_input, scheduled=network_config['scheduled']) # sampled_z(B,C,1,1,T)
        
        if network_config['loss_func'] == 'mmd':
            losses = network.loss_function_mmd(real_img, x_recon, q_z, p_z)
        elif network_config['loss_func'] == 'kld':
            losses = network.loss_function_kld(real_img, x_recon, q_z, p_z)
        else:
            raise ValueError('unrecognized loss function')
        
        losses['loss'].backward()
        
        opti.step()

        loss_meter.update(losses['loss'].detach().cpu().item())
        recons_meter.update(losses['Reconstruction_Loss'].detach().cpu().item())
        dist_meter.update(losses['Distance_Loss'].detach().cpu().item())

        mean_q_z = (q_z.mean(0).detach().cpu() + batch_idx * mean_q_z) / (batch_idx+1) # (C,k,T)
        mean_p_z = (p_z.mean(0).detach().cpu() + batch_idx * mean_p_z) / (batch_idx+1) # (C,k,T)
        mean_sampled_z = (sampled_z.mean(0).detach().cpu() + batch_idx * mean_sampled_z) / (batch_idx+1) # (C,T)

        print(f'Direct_Encoding: {direct_encoding} Train[{epoch}/{max_epoch}] [{batch_idx}/{len(trainloader)}] Loss: {loss_meter.avg}, RECONS: {recons_meter.avg}, DISTANCE: {dist_meter.avg}')

        if batch_idx == len(trainloader)-1:
            os.makedirs(f'checkpoint/{args.name}/imgs/train/', exist_ok=True)
            torchvision.utils.save_image((noise_img+1)/2, f'checkpoint/{args.name}/imgs/train/epoch{epoch}_noise.png')
            torchvision.utils.save_image((real_img+1)/2, f'checkpoint/{args.name}/imgs/train/epoch{epoch}_input.png')
            torchvision.utils.save_image((x_recon+1)/2, f'checkpoint/{args.name}/imgs/train/epoch{epoch}_recons.png')
            writer.add_images('Train/input_img', (real_img+1)/2, epoch)
            writer.add_images('Train/noise_img', (noise_img+1)/2, epoch)
            writer.add_images('Train/recons_img', (x_recon+1)/2, epoch)

    logging.info(f"Direct_Encoding: {direct_encoding} Train [{epoch}] Loss: {loss_meter.avg} ReconsLoss: {recons_meter.avg} DISTANCE: {dist_meter.avg}")
    writer.add_scalar('Train/loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/recons_loss', recons_meter.avg, epoch)
    writer.add_scalar('Train/distance', dist_meter.avg, epoch)
    writer.add_scalar('Train/mean_q', mean_q_z.mean().item(), epoch)
    writer.add_scalar('Train/mean_p', mean_p_z.mean().item(), epoch)
    

    writer.add_image('Train/mean_sampled_z', mean_sampled_z.unsqueeze(0), epoch)
    mean_q_z = mean_q_z.permute(1,0,2) # (k,C,T)
    mean_p_z = mean_p_z.permute(1,0,2) # (k,C,T)
    writer.add_image(f'Train/mean_q_z', mean_q_z.mean(0).unsqueeze(0))
    writer.add_image(f'Train/mean_p_z', mean_p_z.mean(0).unsqueeze(0))

    return loss_meter.avg


def test(network, testloader, epoch, direct=True):
    n_steps = glv.network_config['n_steps']
    max_epoch = glv.network_config['epochs']

    loss_meter = AverageMeter()
    recons_meter = AverageMeter()
    dist_meter = AverageMeter()

    mean_q_z = 0
    mean_p_z = 0
    mean_sampled_z = 0

    count_mul_add, hook_handles = add_hook(net)

    network = network.eval()
    with torch.no_grad():
        for batch_idx, (real_img, labels) in enumerate(testloader):  
            noise_img= impaint(real_img)
            noise_img = noise_img.to(init_device, non_blocking=True)
            real_img = real_img.to(init_device, non_blocking=True)
            labels = labels.to(init_device, non_blocking=True)
            # direct spike input or rate_based encoding
            if direct == 1:
                direct_encoding = 1
                spike_input = noise_img.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps) # (N,C,H,W,T)
            else: 
                direct_encoding = 0
                spike_input = transformRate(noise_img, n_steps, 0.3, seed=0).to(init_device, non_blocking=True)

            x_recon, q_z, p_z, sampled_z = network(spike_input, scheduled=network_config['scheduled'])

            if network_config['loss_func'] == 'mmd':
                losses = network.loss_function_mmd(real_img, x_recon, q_z, p_z)
            elif network_config['loss_func'] == 'kld':
                losses = network.loss_function_kld(real_img, x_recon, q_z, p_z)
            else:
                raise ValueError('unrecognized loss function')

            mean_q_z = (q_z.mean(0).detach().cpu() + batch_idx * mean_q_z) / (batch_idx+1) # (C,k,T)
            mean_p_z = (p_z.mean(0).detach().cpu() + batch_idx * mean_p_z) / (batch_idx+1) # (C,k,T)
            mean_sampled_z = (sampled_z.mean(0).detach().cpu() + batch_idx * mean_sampled_z) / (batch_idx+1) # (C,T)
            
            loss_meter.update(losses['loss'].detach().cpu().item())
            recons_meter.update(losses['Reconstruction_Loss'].detach().cpu().item())
            dist_meter.update(losses['Distance_Loss'].detach().cpu().item())

            print(f'Direct_Encoding: {direct_encoding} Test[{epoch}/{max_epoch}] [{batch_idx}/{len(testloader)}] Loss: {loss_meter.avg}, RECONS: {recons_meter.avg}, DISTANCE: {dist_meter.avg}')

            if batch_idx == len(testloader)-1:
                os.makedirs(f'checkpoint/{args.name}/imgs/test/', exist_ok=True)
                torchvision.utils.save_image((real_img+1)/2, f'checkpoint/{args.name}/imgs/test/epoch{epoch}_input.png')
                torchvision.utils.save_image((noise_img+1)/2, f'checkpoint/{args.name}/imgs/test/epoch{epoch}_noise.png')
                torchvision.utils.save_image((x_recon+1)/2, f'checkpoint/{args.name}/imgs/test/epoch{epoch}_recons.png')
                writer.add_images('Test/input_img', (real_img+1)/2, epoch)
                writer.add_images('Test/recons_img', (x_recon+1)/2, epoch)
                writer.add_images('Test/noise_img', (noise_img+1)/2, epoch)
                

    logging.info(f"Direct_Encoding: {direct_encoding} Test [{epoch}] Loss: {loss_meter.avg} ReconsLoss: {recons_meter.avg} DISTANCE: {dist_meter.avg}")
    writer.add_scalar('Test/loss', loss_meter.avg, epoch)
    writer.add_scalar('Test/recons_loss', recons_meter.avg, epoch)
    writer.add_scalar('Test/distance', dist_meter.avg, epoch)
    writer.add_scalar('Test/mean_q', mean_q_z.mean().item(), epoch)
    writer.add_scalar('Test/mean_p', mean_p_z.mean().item(), epoch)
    writer.add_scalar('Test/mul', count_mul_add.mul_sum.item() / len(testloader), epoch)
    writer.add_scalar('Test/add', count_mul_add.add_sum.item() / len(testloader), epoch)
    
    for handle in hook_handles:
        handle.remove()

    writer.add_image('Test/mean_sampled_z', mean_sampled_z.unsqueeze(0), epoch)
    mean_q_z = mean_q_z.permute(1,0,2) # # (k,C,T)
    mean_p_z = mean_p_z.permute(1,0,2) # # (k,C,T)
    writer.add_image(f'Test/mean_q_z', mean_q_z.mean(0).unsqueeze(0))
    writer.add_image(f'Test/mean_p_z', mean_p_z.mean(0).unsqueeze(0))

    return loss_meter.avg

def sample(network, epoch, batch_size=128):
    network = network.eval()
    with torch.no_grad():
        sampled_x, sampled_z = network.sample(batch_size)
        writer.add_images('Sample/sample_img', (sampled_x+1)/2, epoch)
        writer.add_image('Sample/mean_sampled_z', sampled_z.mean(0).unsqueeze(0), epoch)
        os.makedirs(f'checkpoint/{args.name}/imgs/sample/', exist_ok=True)
        torchvision.utils.save_image((sampled_x+1)/2, f'checkpoint/{args.name}/imgs/sample/epoch{epoch}_sample.png')

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', default='debug', type=str)
    parser.add_argument('-config', default='NetworkConfigs/MNIST.yaml', action='store', dest='config', help='The path of config file')
    parser.add_argument('-checkpoint', action='store', dest='checkpoint', help='The path of checkpoint, if use checkpoint')
    parser.add_argument('-device', type=int, default='0')
    parser.add_argument('-de', type=int, default='1', help='direct encoding or rate encoding')


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.config is None:
        raise Exception('Unrecognized config file.')

    if args.device is None:
        init_device = torch.device("cuda:0")
    else:
        init_device = torch.device(f"cuda:{args.device}")
    
    os.makedirs(f'checkpoint/{args.name}', exist_ok=True)
    writer = SummaryWriter(log_dir=f'checkpoint/{args.name}/tb')
    logging.basicConfig(filename=f'checkpoint/{args.name}/{args.name}.log', level=logging.INFO)
    
    logging.info("start parsing settings")
    
    params = parse(args.config)
    network_config = params['Network']
    
    logging.info("finish parsing settings")
    logging.info(network_config)
    print(network_config)
    
    # Check whether a GPU is available
    # if torch.cuda.is_available():
    #     cuda.init()
    #     c_device = aboutCudaDevices()
    #     print(c_device.info())
    #     print("selected device: ", args.device)
    # else:
    #     raise Exception("only support gpu")

    
    glv.init(network_config, [args.device])

    dataset_name = glv.network_config['dataset']
    data_path = glv.network_config['data_path']
    
    logging.info("dataset loading...")
    if dataset_name == "MNIST":
        data_path = os.path.expanduser(data_path)
        train_loader, test_loader = load_dataset_snn.load_mnist(data_path)
    else:
        raise Exception('Unrecognized dataset name.')
    logging.info("dataset loaded")



    if network_config['model'] == 'FSVAE':
        net = fsvae.FSVAE()
    elif network_config['model'] == 'FSVAE_large':
        net = fsvae.FSVAELarge()
    elif network_config['model'] == 'Pruning_FSVAE':
        net = fsvae.Pruning_FSVAE()
    elif network_config['model'] == 'FSVAE_fashion': # fashion mnist ANN
        net = fsvae.FSVAE_fashion()
    else:
        raise Exception('not defined model')

    net = net.to(init_device)
    
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint)    
    optimizer = torch.optim.AdamW(net.parameters(), 
                                lr=glv.network_config['lr'], 
                                betas=(0.9, 0.999), 
                                weight_decay=0.001)
    
    best_loss = 1e8

    for e in range(glv.network_config['epochs']):
        #write_weight_hist(net, e)
        if network_config['scheduled']:
            net.update_p(e, glv.network_config['epochs'])
            logging.info("update p")
        train_loss = train(net, train_loader, optimizer, e, direct=args.de,)
        test_loss = test(net, test_loader, e, direct=args.de)

        torch.save(net.state_dict(), f'checkpoint/{args.name}/checkpoint.pth')
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(net.state_dict(), f'checkpoint/{args.name}/best.pth')
        logging.info(f"Best Loss: {best_loss}")

        sample(net, e, batch_size=128)
        # calc_inception_score(net, e)
        # calc_autoencoder_frechet_distance(net, e)
        # calc_clean_fid(net, e)
        
    writer.close()

