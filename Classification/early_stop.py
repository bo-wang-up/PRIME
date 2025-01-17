import argparse
import logging
import os
import glob
import time
import sys
import yaml

import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from network import *

# read neuromorphic datasets
from spikingjelly.datasets.n_mnist import NMNIST

import tonic

# Training settings
parser = argparse.ArgumentParser(description='PyTorch NMNIST')
parser.add_argument('-name',           type=str,       default='main',          help='experiment network name')
parser.add_argument('-dataset',        type=str,       default='nmnist',          help='dataset')
parser.add_argument('-net',            type=str,       default='NMNIST_CONV',             help='architecture of network')
parser.add_argument('-mode',           type=str,       default='CONV',              help='the backbone for the task')
parser.add_argument('-data',           type=str,       default='/home/wangbo/dataset/NMNIST/', help='location of the data corpus')
parser.add_argument('-test_batch_size',type=int,       default=1,               help='input batch size for testing (default: 1000)')
parser.add_argument('-lr',             type=float,     default=1e-3,               help='init inner learning rate')
parser.add_argument('-momentum',       type=float,     default=0.9,                help='momentum of SGD')
parser.add_argument('-report_freq',    type=float,     default=100,                 help='report frequency')
parser.add_argument('-gpu',            type=str,       default='0',                help='gpu device id')
parser.add_argument('-seed',           type=int,       default=1,                  help='random seed')
parser.add_argument('-Ns',             type=int,       default=10,                 help='number of timestamps')
parser.add_argument('-comments',       type=str,       default='software',    help='comments of exp')
args = parser.parse_args()

if not os.path.exists('record/weight_test/ee'):
    os.makedirs('record/weight_test/ee')
save_path = 'record/weight_test/ee/Test-{}-{}-{}'.format(args.dataset, args.net, args.comments)
utils.create_exp_dir(save_path, scripts_to_save=glob.glob(args.name + '.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.getLogger().setLevel(logging.INFO)

# writer = SummaryWriter('./Pruning_SNN/tb_save')


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = ' + args.gpu)
    logging.info("args = %s", args)

    print("load dataset:")
    if args.dataset == 'nmnist':
        train_dst = NMNIST(args.data, train=True, data_type='frame', split_by='number', frames_number=args.Ns)
        test_dst = NMNIST(args.data, train=False, data_type='frame', split_by='number', frames_number=args.Ns)
    if args.dataset == 'pokerdvs':

        sensor_size = tonic.datasets.POKERDVS.sensor_size
        frame_transform = tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=10)

        train_dst = tonic.datasets.POKERDVS(save_to=args.data,transform=frame_transform, train=True)
        test_dst = tonic.datasets.POKERDVS(save_to=args.data, transform=frame_transform, train=False)

    test_queue = DataLoader(
        test_dst,
        batch_size=1,
        shuffle=True,
        # num_workers=4,
        pin_memory=True,
    )

    print("load model:")
    arch = yaml.safe_load(open('arch.yaml', mode='r', encoding='utf-8'))
    criterion = nn.CrossEntropyLoss().cuda()


    if args.mode == 'MLP':
        model = LinearNet(
            fc_size=arch[args.net]['fc_size'],
            sparsity = args.sparsity
        )
    elif args.mode == 'CONV':
        model = ConvNet_ee(
            conv_size = arch[args.net]['conv_size'],
            fc_size = arch[args.net]['fc_size'],
            pooling_pos = arch[args.net]['pooling_pos'],
            sparsity = 0.5,
            criterion=criterion
        )


    model = model.cuda()
    model.load_state_dict(torch.load('/home/wangbo/codes/pruning_example_code/record/TRAIN-nmnist-NMNIST_CONV-20250117-192757-train/params.pt'))
    print(next(model.parameters()).device)

    acc = test_early_exit(model, test_queue)
        




def test_early_exit(model, test_loader):
    model.eval()
    accuracy = 0
    sum_acc = 0
    sum_timestep = 0
    
    confidence1 = 0
    confidence2 = 0
    sum_confidence1 = 0
    sum_confidence2 = 0

    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    gtlist=torch.zeros(0,dtype=torch.long, device='cpu')

    labels = []
    output_feature = []

    for batch_idx, (input, target) in enumerate(test_loader):
        labels.append(target)
        with torch.no_grad():
            if input.type() == 'torch.ShortTensor':
                input = input.float()
            input, target = input.cuda(), target.cuda()

            output, timestep, confidence1, confidence2= model(input, args.Ns)

            output_feature.append(output)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            accuracy = pred.eq(target.view_as(pred)).sum().item() * 100

            predlist=torch.cat([predlist,pred.view(-1).cpu()])
            gtlist=torch.cat([gtlist,target.view(-1).cpu()])

            sum_acc += accuracy
            sum_timestep += timestep

            sum_confidence1 += confidence1
            sum_confidence2 += confidence2

            if (batch_idx+1) % args.report_freq == 0:
                logging.info('[{}/{} ({:.0f}%)]\tAcc:{:.3f}\tts:{:.4f}'.format(
                    (batch_idx+1) * len(input), len(test_loader.dataset),
                    100. * (batch_idx+1) / len(test_loader), 
                    sum_acc/((batch_idx+1) * len(input)), 
                    sum_timestep/((batch_idx+1) * len(input))
                    ))                

    # conf_mat = confusion_matrix(gtlist.numpy(), predlist.numpy())
    # logging.info(conf_mat)

    # class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    # logging.info(class_accuracy)

    # np.save(os.path.join(save_path, 'confusion_matrix.npy'), conf_mat)

    # torch.save(model.layer0_feature, os.path.join(save_path, 'layer0_feature.pt'))
    # torch.save(model.layer1_feature, os.path.join(save_path, 'layer1_feature.pt'))
    # torch.save(model.flatten_feature, os.path.join(save_path, 'flatten_feature.pt'))

    # labels = torch.cat(labels, dim=0)
    # torch.save(labels, os.path.join(save_path, 'labels.pt'))  

    # output_feature = torch.cat(output_feature, dim=0)
    # torch.save(output_feature, os.path.join(save_path, 'output_feature.pt'))

    sum_acc = sum_acc / len(test_loader.dataset)
    sum_timestep /= len(test_loader.dataset)


    sum_confidence1 /= len(test_loader.dataset)
    sum_confidence2 /= len(test_loader.dataset)

    logging.info('\nTest set: Average timestep: {:.4f}, Accuracy: {:.4f}%'.format(
        sum_timestep, sum_acc))
    
    logging.info('\nTest set: Average confidence1: {:.4f}, confidence2: {:.4f}'.format(
        sum_confidence1, sum_confidence2)) 
    
if __name__ == '__main__':
    main()