from layer import *

from spikingjelly.clock_driven import layer as cdlayer

class ConvNet(nn.Module):
    def __init__(self, conv_size, fc_size, pooling_pos, sparsity, criterion=nn.CrossEntropyLoss()):
        super(ConvNet, self).__init__()
        # arch parameters
        self.conv_num = len(conv_size)
        self.fc_num = len(fc_size) - 1
        self.criterion = criterion
        self.sparsity = sparsity


        # conv layers
        self.features = nn.Sequential()

        for i in range(self.conv_num):
            in_channel, out_channel, kernel, stride, padding = eval(conv_size[i])
            layer = Conv_Cell(in_channel, out_channel, kernel, stride, padding, self.sparsity)
            self.features.add_module('conv{}'.format(i), layer)

            if i in pooling_pos:
                # default pooling: average pooling
                self.features.add_module('avgpool{}'.format(i), nn.AvgPool2d(2, 2))

            # #dropout layer
            # self.features.add_module('dropout2d'.format(i), Dropout2d(0.03))

        # fc layers
        self.classifier = nn.Sequential()

        for i in range(self.fc_num):
            input_size = fc_size[i]
            hidden_size = fc_size[i + 1]
            if i < self.fc_num - 1:
                layer = Linear_Cell(input_size, hidden_size, self.sparsity)
            else:
                layer = Output_Cell(input_size, hidden_size, self.sparsity)

            self.classifier.add_module('linear{}'.format(i), layer)
        # for i in range(self.fc_num):
        #     input_size = fc_size[i]
            
        #     hidden_size = fc_size[i + 1]
        #     layer = Linear_Cell(input_size, hidden_size, self.sparsity)

        #     self.classifier.add_module('linear{}'.format(i), layer)


    def _reset(self):
        for layer in self.modules():
            if isinstance(layer, MemoryUnit):
                layer.reset()

    def forward(self, input: torch.FloatTensor, timestamps: int ):

        self._reset()

        outputs = []
        for t in range(timestamps):
            x = input[:, t, :] if len(input.size()) == 3 else input[:, t, :, :, :]
            x = self.features(x)
            x = torch.flatten(x, start_dim=1)
            x = self.classifier(x)
            outputs.append(x.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1).contiguous()
        output = torch.mean(outputs, dim=1)

        return output, outputs


    def accuracy(self, output: torch.FloatTensor, target: torch.FloatTensor, bs: int):
        # return torch.mean(torch.eq(target, torch.argmax(output, 1)).float())
        accuracy = 0
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        accuracy += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * accuracy / bs
        return accuracy

    def loss(self, output: torch.FloatTensor, target: torch.FloatTensor):
        return self.criterion(output, target)
    
    def TET_loss(self, outputs, target, means = 0.5, lamb = 1e-3):
        T = outputs.size(1)
        Loss_es = 0
        for t in range(T):
            Loss_es += self.criterion(outputs[:, t, ...], target)
        Loss_es = Loss_es / T # L_TET
        if lamb != 0:
            MMDLoss = torch.nn.MSELoss()
            y = torch.zeros_like(outputs).fill_(means)
            Loss_mmd = MMDLoss(outputs, y) # L_mse
        else:
            Loss_mmd = 0
        return (1 - lamb) * Loss_es + lamb * Loss_mmd # L_Total
    

#########
##weight_train_ee
#########
class ConvNet_ee(ConvNet):
    def __init__(self, conv_size, fc_size, pooling_pos, sparsity, criterion=nn.CrossEntropyLoss()):
        super(ConvNet_ee, self).__init__(conv_size=conv_size,
                         fc_size=fc_size,
                         pooling_pos=pooling_pos,
                         sparsity=sparsity,
                         criterion=criterion,
                         )
        self.output = None
        
    def forward(self, input: torch.FloatTensor, timestamps: int):

        self._reset()
        
        outputs = []

        confidence_1stmax = 0
        confidence_2ndmax = 0
        eet = 0

        for t in range(timestamps):

            x = input[:, t, :] if len(input.size()) == 3 else input[:, t, :, :, :]
            x = self.features(x)
            x = torch.flatten(x, start_dim=1)
            x = self.classifier(x)

            outputs.append(x.unsqueeze(1))
            outputs_tensor = torch.cat(outputs, dim=1).contiguous()
            output = torch.mean(outputs_tensor, dim=1)


            # confidence score
            alfha = 2 # regulation parameter, prevent the saturated confidence
            output_ee = output/alfha

            softmaxes  = F.softmax(output_ee, dim=1)
            confidences= torch.topk(softmaxes, 2)
            eet = t

            confidence_1stmax = float(confidences.values[0][0].item())
            confidence_2ndmax = float(confidences.values[0][1].item())

            #########
            # Early Exit Policy 
            ##########
            ee_threshold = 0.8
            if confidences.values[0][0].item() > ee_threshold:
                break


        return output, eet+1, confidence_1stmax, confidence_2ndmax
    


class LinearNet(nn.Module):
    def __init__(self, fc_size, sparsity, criterion=nn.CrossEntropyLoss()):
        """
        :param fc_size: shape of full-connected layers
        :param criterion: criterion that measures the accuracy
        """
        super(LinearNet, self).__init__()
        # arch parameters
        self.fc_num = len(fc_size) - 1
        self.criterion = criterion
        self.sparsity = sparsity

        # fc layers
        self.classifier = nn.Sequential()

        for i in range(self.fc_num):
            input_size = fc_size[i]
            hidden_size = fc_size[i + 1]
            if i < self.fc_num - 1:
                layer = Linear_Cell(input_size, hidden_size, self.sparsity)
            else:
                layer = Output_Cell(input_size, hidden_size, self.sparsity)

            self.classifier.add_module('linear{}'.format(i), layer)

        # for i in range(self.fc_num):
        #     input_size = fc_size[i]
            
        #     hidden_size = fc_size[i + 1]
        #     layer = Linear_Cell(input_size, hidden_size, self.sparsity)

        #     self.classifier.add_module('linear{}'.format(i), layer)


    def _reset(self):
        for layer in self.modules():
            if isinstance(layer, MemoryUnit):
                layer.reset()

    def forward(self, input: torch.FloatTensor, timestamps: int ):

        self._reset()

        outputs = []
        for t in range(timestamps):
            x = input[:, t, :]
            x = self.classifier(x)
            outputs.append(x.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1).contiguous()
        avg_potential = torch.mean(outputs, dim=1)

        return avg_potential, outputs



    def accuracy(self, output: torch.FloatTensor, target: torch.FloatTensor, bs: int):
        # return torch.mean(torch.eq(target, torch.argmax(output, 1)).float())
        accuracy = 0
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        accuracy += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * accuracy / bs
        return accuracy

    def loss(self, input: torch.FloatTensor, target: torch.FloatTensor):
        return self.criterion(input, target)
