import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_CIFAR10_dataset():
    transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
    testset =  torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)
    return trainset, testset
