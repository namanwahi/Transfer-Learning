import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def CIFAR10_whole():
    return whole_trainset, whole_testset

def CIFAR10_subset_sampler(class_indices):
    return SubsetRandomSampler(class_indices)

def CIFAR10_animal_sampler():
    return CIFAR10_subset_sampler([2, 3, 4, 5, 6, 7])
    

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

whole_trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)

whole_testset =  torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)


