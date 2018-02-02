import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import os

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

resnet_18_default = 224

def _get_dataset(resize=resnet_18_default):
    transform = transforms.Compose(
    [transforms.Resize(resize),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=dir_path, train=True, download=True, transform=transform)
    testset =  torchvision.datasets.CIFAR10(root=dir_path, train=False, download=True, transform=transform)
    return trainset, testset

def _get_classes():
    return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_dataloader(train=True, batch_size=16):
    animal_indices = [2, 3, 4, 5, 6, 7]
    #animal_sampler = SubsetRandomSampler(animal_indices)
    if train:
        return DataLoader(_get_dataset()[0], batch_size)
    else:
        return DataLoader(_get_dataset()[1], batch_size)
        

