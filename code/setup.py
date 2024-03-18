import random
import torch
import torch.nn as nn

from torchvision import models

def create_networks():
    resnet152 = nn.Sequential(*list(models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1).children())[:-1])
    torch.save(resnet152.state_dict(), './resnet152.pth')

    resnet18 = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).children())[:-1])
    torch.save(resnet18.state_dict(), './resnet18.pth')

    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
    torch.save(vgg16.state_dict(), './vgg16.pth')

    vgg11 = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).features
    torch.save(vgg11.state_dict(), './vgg11.pth')

    resnext = nn.Sequential(*list(models.resnext50_32x4d(weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1).children())[:-1])
    torch.save(resnext.state_dict(), './resnext50.pth')
    
    swin = nn.Sequential(*list(models.swin_s(weights = models.Swin_S_Weights.IMAGENET1K_V1).children())[:-3])
    torch.save(swin.state_dict(), './swintransformer.pth')
    
def main() -> None:
    random.seed(42)
    create_networks()


if __name__ == "__main__":
    main()
