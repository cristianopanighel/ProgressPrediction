import random
import torch
import torch.nn as nn

from torchvision import models

def create_networks():
    resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    torch.save(resnet.state_dict(), './resnet152.pth')

    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    torch.save(resnet.state_dict(), './resnet18.pth')

    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
    torch.save(vgg16.state_dict(), './vgg16.pth')

    vgg11 = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).features
    torch.save(vgg11.state_dict(), './vgg11.pth')


def main() -> None:
    random.seed(42)
    create_networks()


if __name__ == "__main__":
    main()
