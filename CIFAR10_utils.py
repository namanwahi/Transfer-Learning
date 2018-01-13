import torch
import torchvision
import torchvision.transforms as transforms

def CIFAR10_whole():
    return whole_trainset, whole_testset

def _CIFAR10_subset(class_indices):
    trainset = [x for x in whole_trainset if x[1] in class_indices]
    testset = [x for x in whole_testset if x[1] in class_indices]
    return trainset, testset

def CIFAR_animals():
    return _CIFAAR10_subset([2, 3, 4, 5, 6, 7])
    

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

whole_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

whole_testset =  torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train, test = _CIFAR10_subset([1,2,3])

