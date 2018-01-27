import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

def _get_dataset():
    transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
    testset =  torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)
    return trainset, testset

def _get_classes():
    return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_training_dataloader():
    animal_indices = [2, 3, 4, 5, 6, 7]
    animal_sampler = SubsetRandomSamples(animal_indices)
    return DataLoader(_get_dataset()[0], batch_size=256, sampler=animal_sampler)


