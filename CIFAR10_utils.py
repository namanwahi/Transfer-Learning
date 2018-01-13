import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

whole_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

whole_testset =  torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
