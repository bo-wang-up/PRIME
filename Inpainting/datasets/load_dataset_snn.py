import os
import torchvision.datasets
import torchvision.transforms as transforms
import torch
import global_v as glv
import numpy as np

def load_mnist(data_path):
    print("loading MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])
    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

def load_mini_mnist(data_path, train_num=1000, test_num=1000):
    print("loading MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])
    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform_test, download=True)
    # evenly select samples from each class
    sample_num = test_num
    np.random.seed(0)
    y_test = np.array([testset[i][1] for i in range(len(testset))])
    class_indices = [np.where(y_test == i)[0] for i in range(10)]
    sample_indices = [np.random.choice(class_indices[i], size=sample_num, replace=False) for i in range(10)]
    sample_indices = np.array(sample_indices).reshape(-1)

    testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, sample_indices),
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=2, 
                            pin_memory=True)
    # evenly select samples from each class
    sample_num = train_num
    np.random.seed(0)
    y_train = np.array([trainset[i][1] for i in range(len(trainset))])
    class_indices = [np.where(y_train == i)[0] for i in range(10)]
    sample_indices = [np.random.choice(class_indices[i], size=sample_num, replace=False) for i in range(10)]
    sample_indices = np.array(sample_indices).reshape(-1)
    trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, sample_indices),
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=2, 
                        pin_memory=True)
    return trainloader, testloader
    


def load_fashionmnist(data_path):
    print("loading Fashion MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = torchvision.datasets.FashionMNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.FashionMNIST(data_path, train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    return trainloader, testloader

def load_fashionmnist_mini(data_path, test_num=100):
    print("loading Fashion MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])
    testset = torchvision.datasets.FashionMNIST(data_path, train=False, transform=transform_test, download=True)
    # evenly select samples from each class
    sample_num = test_num
    np.random.seed(0)
    y_test = np.array([testset[i][1] for i in range(len(testset))])
    class_indices = [np.where(y_test == i)[0] for i in range(10)]
    sample_indices = [np.random.choice(class_indices[i], size=sample_num, replace=False) for i in range(10)]
    sample_indices = np.array(sample_indices).reshape(-1)

    testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, sample_indices),
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=2, 
                            pin_memory=True)
    return testloader

def load_cifar10(data_path):
    print("loading CIFAR10")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = torchvision.datasets.CIFAR10(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.CIFAR10(data_path, train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
    return trainloader, testloader

def load_celebA(data_path):
    print("loading CelebA")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']
    
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange])

    trainset = torchvision.datasets.CelebA(root=data_path, 
                                            split='train', 
                                            download=True, 
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=batch_size, 
                                            shuffle=True, num_workers=8, pin_memory=True)

    testset = torchvision.datasets.CelebA(root=data_path, 
                                            split='test', 
                                            download=True, 
                                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size, 
                                            shuffle=False, num_workers=8, pin_memory=True)
    return trainloader, testloader



