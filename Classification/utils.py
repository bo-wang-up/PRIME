import numpy as np
import shutil
import torch
import os
from torch.distributions.bernoulli import Bernoulli


def count_parameters_in_MB(model_param):
    return np.sum(np.prod(p.size()) for p in model_param if p.requires_grad) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def save_scores(model, model_path):
    for i in model.scores:
        i = i.cpu().detach()
    torch.save(model.scores, model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def format_name(model_path):
    state_dict = torch.load(model_path)
    old_list = []

    for k in state_dict.keys():
        if k.startswith('attr_param') or k.startswith('nonattr_param'):
            old_list.append(k)

    for k in old_list:
        if k.startswith('attr_param'):
            state_dict[k.replace('attr_param', 'attributes')] = state_dict.pop(k)
        elif k.startswith('nonattr_param'):
            state_dict[k.replace('nonattr_param', 'weights')] = state_dict.pop(k)

    torch.save(state_dict, model_path)

    return state_dict


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# Add Gaussian noise during transform
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1, seed=0):
        self.std = std
        self.mean = mean

        torch.manual_seed(seed)

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# Add salt and pepper noise during transform
class AddSaltAndPepperNoise(object):
    def __init__(self, amount=0.3, seed=0):
        self.amount = amount

        torch.manual_seed(seed)

    def __call__(self, tensor: torch.Tensor):
        flipped = Bernoulli(torch.full(tensor.size(), self.amount)).sample().bool()
        salted = Bernoulli(torch.full(tensor.size(), 0.5)).sample().bool()
        peppered = ~salted

        tensor = torch.where(flipped & salted, torch.ones_like(tensor), tensor)
        tensor = torch.where(flipped & peppered, torch.zeros_like(tensor), tensor)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(amount={0})'.format(self.amount,)


# Add speckle noise during transform
class AddSpeckleNoise(object):
    def __init__(self, mean=0., std=0.1, seed=0):
        self.std = std
        self.mean = mean

        torch.manual_seed(seed)

    def __call__(self, tensor):
        return tensor + tensor * (torch.randn(tensor.size()) * self.std + self.mean)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



