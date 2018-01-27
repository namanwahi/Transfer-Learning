import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def CIFAR10_whole():
    transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    whole_trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
    whole_testset =  torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)
    return whole_trainset, whole_testset

