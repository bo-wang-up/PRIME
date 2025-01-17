import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from copy import deepcopy

lens =  0.5 # default 1.0

software_scale = 0.01

def kaiming_normalize(tensor, a=0, model='fan_in', nonlinearity='relu'):
    fan = nn.init._calculate_correct_fan(tensor, model)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return (tensor - 0) / std


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # input = u - Vth, if input > 0, output 1
        output = torch.gt(input, 0.)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #fu = torch.tanh(input)
        #fu = 1 - torch.mul(fu, fu)
        fu = abs(input) < lens
        fu = fu / (2 * lens)

        return grad_input * fu.float()

spikeplus = STEFunction.apply

# class GetSubnet(autograd.Function):
#     @staticmethod
#     def forward(ctx, scores, k):
#         # Get the supermask by sorting the scores and using the top k%
#         out = scores.clone()
#         _, idx = scores.flatten().sort()
#         j = int((1 - k) * scores.numel())

#         # flat_out and out access the same memory.
#         flat_out = out.flatten()
#         flat_out[idx[:j]] = 0
#         flat_out[idx[j:]] = 1

#         return out

#     @staticmethod
#     def backward(ctx, g):
#         # send the gradient g straight-through on the backward pass.
#         return g, None

def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()
    
class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        k_val = percentile(scores, sparsity*100)
        return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None
    
class MemoryUnit(nn.Module):
    def __init__(self):
        """
        The base class of memory cells in the network,
        of which the reset method needs to be overridden so that the network resets its state variables
        (such as U and s in the cell, mask in the dropout layer) at the beginning of each simulation.
        """
        super(MemoryUnit, self).__init__()

    def reset(self):
        raise NotImplementedError


class Dropout(MemoryUnit):
    def __init__(self, p):
        """
        The implementation of Dropout in SNN,
        which has almost the same function as torch.nn.Dropout,
        but keeping the mask unchanged in each timestep and updating it manually.
        :param p: probability of an element to be zeroed
        """
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def reset(self):
        self.mask = None

    def _generate_mask(self, input):
        self.mask = F.dropout(torch.ones_like(input.data), self.p, training=True)

    def forward(self, input):
        if self.training:
            if self.mask is None:
                self._generate_mask(input)
            out = torch.mul(input, self.mask)
        else:
            out = input
        return out


class Dropout2d(Dropout):
    def __init__(self, p):
        """
        Same as Dropout implementation in SNN,
        but in the form of 2D.
        :param p: probability of an element to be zeroed
        """
        super().__init__(p)

    def _generate_mask(self, input):
        self.mask = F.dropout2d(torch.ones_like(input.data), self.p, training=True)


class Basic_Cell(MemoryUnit):
    def __init__(self, weight_size, attr_size, sparsity, activation=spikeplus):
        super(Basic_Cell, self).__init__()
        self._g = nn.LeakyReLU(0.1)
        self._activation = activation
        self.sparsity = sparsity

        # define trainable scores
        self.scores = nn.Parameter(torch.Tensor(weight_size))
        # nn.init.kaiming_normal_(self.scores, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # define untrainable weight
        self.weight = nn.Parameter(torch.Tensor(weight_size), requires_grad=False)
        nn.init.normal_(self.weight, mean=0, std=6.352) # normal distribution similar to rram

        self.zeros = torch.zeros_like(self.scores)
        self.ones = torch.ones_like(self.scores)


        self.th = 0.5
        self.decay = 0.8
        

        # define the state variable
        self.U = torch.Tensor([0.]).cuda()
        self.s = torch.Tensor([0.]).cuda()

    def reset(self):
        self.U = torch.Tensor([0.]).cuda()
        self.s = torch.Tensor([0.]).cuda()

    def ops(self, input: torch.FloatTensor):
        raise NotImplementedError
    

    def forward(self, input: torch.FloatTensor):
        
        self.U = torch.mul(self.U, 1. - self.s) # reset


        self.U = torch.mul(self.U, 1. - self.decay)  # decay

        I = self.ops(input)
        self.U = torch.add(self.U, I)

        # apply the nonlinear function
        self.U = self._g(self.U)

        self.s = self._activation(self.U - self.th)  # calculate the spike
        return self.s


class Linear_Cell(Basic_Cell):
    def __init__(self, input_size, hidden_size, sparsity):
        super(Linear_Cell, self).__init__(
            weight_size=torch.Size([hidden_size, input_size]),
            attr_size=torch.Size([hidden_size]),
            sparsity = sparsity
        )

        self.th = 0.5


    def ops(self, input: torch.FloatTensor):
        subnet = GetSubnet.apply(self.scores.abs(), self.zeros, self.ones, self.sparsity)
        weight = self.weight * software_scale
        w = weight * subnet
        return F.linear(input, w, bias=None)


class Output_Cell(Linear_Cell):
    def __init__(self, input_size, hidden_size, sparsity):
        super(Output_Cell, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            sparsity = sparsity
        )

    def forward(self, input: torch.FloatTensor):

        I = self.ops(input)
        self.U = torch.add(self.U, I)


        # apply the nonlinear function
        self.U = self._g(self.U)

        return self.U


class Conv_Cell(Basic_Cell):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, sparsity):
        super(Conv_Cell, self).__init__(
            weight_size=torch.Size([out_channel, in_channel, kernel, kernel]),
            attr_size=torch.Size([out_channel, 1, 1]),
            sparsity = sparsity
        )
        self.stride = stride
        self.padding = padding


        self.th = 0.2

    def ops(self, input: torch.FloatTensor):
        subnet = GetSubnet.apply(self.scores.abs(), self.zeros, self.ones, self.sparsity)
        weight = self.weight * software_scale
        w = weight * subnet
        return F.conv2d(input, w, bias = None, stride=self.stride, padding=self.padding)

# Cell forDVS Gesture 128
# add the batch normalization layer
# add the noise choice 
class Conv_Cell_128(Basic_Cell): 
    def __init__(self, in_channel, out_channel, kernel, stride, padding, sparsity, noise):
        super(Conv_Cell_128, self).__init__(
            weight_size=torch.Size([out_channel, in_channel, kernel, kernel]),
            attr_size=torch.Size([out_channel, 1, 1]),
            sparsity = sparsity
        )
        self.stride = stride
        self.padding = padding

        # init_w = 0.5
        # self.th = nn.Parameter(torch.tensor(init_w, dtype=torch.float))
        self.th = 1.0
        self.bn = nn.BatchNorm2d(out_channel)
        self.noise = noise # when testing, set noise = 1, when training, set noise = 0

    
    def ops(self, input: torch.FloatTensor):
        weight = self.weight.clone()
        if self.noise == 1:
            weight = weight + torch.randn_like(weight) * 0.02 * weight
        subnet = GetSubnet.apply(self.scores.abs(), self.zeros, self.ones, self.sparsity)
        weight = weight * software_scale
        w = weight * subnet
        return self.bn(F.conv2d(input, w, stride=self.stride, padding=self.padding))

class Output_Cell_128(Linear_Cell):
    def __init__(self, input_size, hidden_size, sparsity, noise):
        super(Output_Cell_128, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            sparsity = sparsity
        )
        self.noise = noise
        self.th = 1.0

    def ops(self, input: torch.FloatTensor):
        weight = self.weight.clone()
        if self.noise == 1:
            weight = weight + torch.randn_like(weight) * 0.02 * weight
        subnet = GetSubnet.apply(self.scores.abs(), self.zeros, self.ones, self.sparsity)
        weight = weight * software_scale
        w = weight * subnet
        return F.linear(input, w, bias=None)

    def forward(self, input: torch.FloatTensor):

        I = self.ops(input)
        self.U = torch.add(self.U, I)
        # apply the nonlinear function
        self.U = self._g(self.U)

        return self.U
