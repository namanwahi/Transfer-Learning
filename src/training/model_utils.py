from torchvision import  models
from data.CIFAR10_utils import get_dataloader
from training.training_utils import train_model
import torch.nn as nn
import torch

imagenet_classes = 1000

def get_resnet18_model(class_no=imagenet_classes, fixed_feature_extractor=True):
    model = models.resnet18(pretrained=True)

    #freeze all the weights of the network
    if fixed_feature_extractor:
        for param in model.parameters():
           param.requires_grad = False

    #replace the final layer of the neural network
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, class_no)



def save_model(model, path):
    torch.save(mode.state_dict(), path)

def load_pretrained_resnet18(path, class_no):
    model = get_resnet18_model(class_no=class_no, fixed_feature_extractor=False)
    model.load_state_dict(torch.load(path))
    return model
    
