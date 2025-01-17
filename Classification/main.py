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
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import tonic

# Training settings
parser = argparse.ArgumentParser(description='PyTorch NMNIST')
parser.add_argument('-name',           type=str,       default='main',          help='experiment network name')
parser.add_argument('-dataset',        type=str,       default='nmnist',          help='dataset')
parser.add_argument('-net',            type=str,       default='NMNIST_CONV',             help='architecture of network')
parser.add_argument('-mode',           type=str,       default='CONV',              help='the backbone for the task, CONV for NMNIST, gesture for DVS128Gesture')
parser.add_argument('-data',           type=str,       default='/home/wangbo/dataset/NMNIST/', help='location of the data corpus')
parser.add_argument('-optimizer',      type=str,       default='adam',             help='optimizer for SNN backpropagation')
parser.add_argument('-batch_size',     type=int,       default=128,                help='batch size, 128 for NMNIST, 16 for Gesture')
parser.add_argument('-test_batch_size',type=int,       default=1000,               help='input batch size for testing (default: 1000)')
parser.add_argument('-lr',             type=float,     default=1e-3,               help='init inner learning rate')
parser.add_argument('-momentum',       type=float,     default=0.9,                help='momentum of SGD')
parser.add_argument('-weight_decay',   type=float,     default=0.0005,               help='weight decay')
parser.add_argument('-report_freq',    type=float,     default=100,                 help='report frequency')
parser.add_argument('-gpu',            type=str,       default='0',                help='gpu device id')
parser.add_argument('-epochs',         type=int,       default=14,                 help='num of training epochs, 14 for NMNIST, 300 for Gesture')
parser.add_argument('-seed',           type=int,       default=0,                  help='random seed')
parser.add_argument('-Ns',             type=int,       default=10,                 help='number of timestamps')
parser.add_argument('-sparsity',       type=float,     default=0.5,                help='how sparse is each layer')
parser.add_argument('-comments',       type=str,       default='train',    help='comments of exp')
args = parser.parse_args()

file_exist = 'record'
if not os.path.exists(file_exist):
    os.makedirs(file_exist)
save_path = file_exist + '/TRAIN-{}-{}-{}-{}'.format(args.dataset, args.net, time.strftime("%Y%m%d-%H%M%S"), args.comments)
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
        train_dst = NMNIST(args.data, train=True, data_type='frame', split_by='time', frames_number=args.Ns)
        test_dst = NMNIST(args.data, train=False, data_type='frame', split_by='time', frames_number=args.Ns)

    elif args.dataset == 'pokerdvs':
        sensor_size = tonic.datasets.POKERDVS.sensor_size
        frame_transform = tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=10)
        train_dst = tonic.datasets.POKERDVS(save_to=args.data,transform=frame_transform, train=True)
        test_dst = tonic.datasets.POKERDVS(save_to=args.data, transform=frame_transform, train=False)
    elif args.dataset == 'dvs_gesture':
        train_dst = DVS128Gesture(root=args.data, train=True, data_type='frame', frames_number=args.Ns, split_by='number')
        test_dst = DVS128Gesture(root=args.data, train=False, data_type='frame', frames_number=args.Ns, split_by='number')


    train_queue = DataLoader(
        train_dst,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=4,
        pin_memory=True,
    )
    test_queue = DataLoader(
        test_dst,
        batch_size=args.test_batch_size,
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
        model = ConvNet(
            conv_size = arch[args.net]['conv_size'],
            fc_size = arch[args.net]['fc_size'],
            pooling_pos = arch[args.net]['pooling_pos'],
            sparsity = args.sparsity,
            criterion=criterion
        )
    elif args.mode == 'gesture':
        model = ConvNet_128(
            conv_size = arch[args.net]['conv_size'],
            fc_size = arch[args.net]['fc_size'],
            pooling_pos = arch[args.net]['pooling_pos'],
            sparsity = args.sparsity,
            criterion=criterion,
            noise = 0
        )
        model = nn.DataParallel(model)


    model = model.cuda()
    print(next(model.parameters()).device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            args.lr,
            weight_decay=args.weight_decay,
        )

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max = args.epochs)


    acc = 0
    acc_max = 0
    train_acc = 0 
    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]


        logging.info('epoch %d lr %e', epoch, lr)

        train_acc = train(model, train_queue, optimizer, epoch)
        scheduler.step()

        # validation
        acc = test(model, test_queue)
        
        if acc_max < acc:
            acc_max = acc
            utils.save(model, os.path.join(save_path, 'params.pt'))
            #utils.save_scores(model, os.path.join(save_path, 'scores.pt'))
        
        logging.info('acc_max: {:.3f}'.format(acc_max))


    
    
def train(model, train_loader, optimizer, epoch):
    model.train()
    avg_acc = 0
    sum_acc = 0
    for batch_idx, (input, target) in enumerate(train_loader):
        if args.mode == 'MLP':
            bs, ts, p, w, h = input.size()
            input = input.reshape(bs, ts, p * w * h)
        if input.type() == 'torch.ShortTensor':
            input = input.float()
        input, target = input.cuda(), target.cuda()
        optimizer.zero_grad()
        output, outputs = model(input, args.Ns)
        loss = model.loss(output, target)
        loss.backward()

        optimizer.step()

        accuracy = model.accuracy(output, target, args.batch_size)
        sum_acc += accuracy
        avg_acc = sum_acc / (batch_idx + 1)

        if batch_idx % args.report_freq == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc:{:.3f}'.format(
                epoch, batch_idx * len(input), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), avg_acc))
    return avg_acc


def test(model, test_loader):
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for input, target in test_loader:
            if args.mode == 'MLP':
                bs, ts, p, w, h = input.size()
                input = input.reshape(bs, ts, p * w * h)
            if input.type() == 'torch.ShortTensor':
                input = input.float()
            input, target = input.cuda(), target.cuda()
            output, outputs = model(input, args.Ns)
            test_loss += model.loss(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            accuracy += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, accuracy, len(test_loader.dataset),
        100. * accuracy / len(test_loader.dataset)))
    return 100. * accuracy / len(test_loader.dataset)
    
if __name__ == '__main__':
    main()