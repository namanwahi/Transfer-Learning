import torch
import torchvision
from torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.RandomHorizonalFlip(),
     transforms.RandomVeritcalFlip(),
     transforms.ToTensor(),
     ptransforms.Normaliza((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     
