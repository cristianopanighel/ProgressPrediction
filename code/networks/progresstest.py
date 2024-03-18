import torch
import torch.nn.functional as F

from .pyramidpooling import SpatialPyramidPooling
from networks.swintransformer import SwinTransformer
from torch import nn
from torchvision import models
from torchvision.transforms import v2
from torchvision.ops import roi_align
from typing import List

class ProgressTest(nn.Module):
    def __init__(
        self,
        pooling_layers: List[int],
        roi_size: int,
        dropout_chance: float,
        embed_dim: int,
        finetune: bool,
        backbone: str,
        backbone_path: str = None,
    ) -> None:
        super().__init__()
        shape = 512
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        match backbone:
            case "r2_1":
                self.backbone = nn.Sequential(*list(models.video.r2plus1d_18(weights = models.video.R2Plus1D_18_Weights).children())[:-1])
            case "vgg11":
                self.backbone = models.vgg11().features
            case "resnext50":
                self.backbone = nn.Sequential(*list(models.resnext50_32x4d(weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1).children())[:-1])
                shape = 2048
            case "resnet18":
                self.backbone = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).children())[:-1])
            case "swintransformer":
                self.backbone = nn.Sequential(*list(models.swin_s(weights = models.Swin_S_Weights.IMAGENET1K_V1).children())[:-3])
                # self.backbone = SwinTransformer()
                shape = 768
            case _:
                raise Exception(
                    f"Backbone {backbone} cannot be used for Progressnet")

        for param in self.parameters():
            param.requires_grad = False

             #self.transforms = v2.Compose([
        #    v2.RandomResizedCrop(size=(224, 224), antialias=True),
        #    v2.RandomHorizontalFlip(p=0.5),
        #    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #    ])

        pooling_size = sum(map(lambda x: x**2, pooling_layers))
        self.roi_size = roi_size

        self.spp = SpatialPyramidPooling(pooling_layers)
        self.spp_fc = nn.Linear(shape * pooling_size, embed_dim)
        self.spp_dropout = nn.Dropout(p=dropout_chance)

        self.roi_fc = nn.Linear(shape * (roi_size**2), embed_dim)
        self.roi_dropout = nn.Dropout(p=dropout_chance)

        self.fc7 = nn.Linear(embed_dim * 2, 64)
        self.fc7_dropout = nn.Dropout(p=dropout_chance)
        
        # dim = 64
        # num_heads = 8 
        # self.multihead_attn = nn.MultiheadAttention(dim, num_heads, batch_first = True)
        input_size = 64
        num_layers = 6
        num_heads = 8
        self.model = nn.Transformer(input_size, num_heads, num_layers, batch_first=True)

        if finetune:
            for param in self.parameters():
                param.requires_grad = False
        
        self.fc8 = nn.Linear(64, 1)
        # self.fc8 = nn.Linear(64, 1)

        self.hidden1, self.hidden2 = None, None

    def forward(self, frames: torch.FloatTensor, boxes: torch.Tensor = None) -> torch.FloatTensor:
        # B = batch size, S = sequence legnth, C = channels, H = Height, W = width
        B, S, C, H, W = frames.shape
        num_samples = B * S

        # print(boxes)
        # print(boxes.shape)

        if boxes is None:
            boxes = torch.FloatTensor([0, 0, 224, 224])
            boxes = boxes.repeat(num_samples, 1)
            boxes = boxes.to(self.device)

        frames = frames.reshape(num_samples, C, H, W)
        # frames = frames.reshape(B, C, num_samples, H, W)
        boxes = boxes.reshape(num_samples, 4)

        # frames = self.transforms(frames)

        frames = self.backbone(frames)
        # Based on the backbone, the shape of the frame could be very different

        spp_pooled = self.spp(frames)
        spp_pooled = self.spp_fc(spp_pooled)
        spp_pooled = self.spp_dropout(spp_pooled)
        spp_pooled = torch.relu(spp_pooled)

        indices = torch.arange(
            0, num_samples, device=self.device).reshape(num_samples, -1)
        boxes = torch.cat((indices, boxes), dim=-1)

        roi_pooled = roi_align(
            frames, boxes, self.roi_size, spatial_scale=0.03125)
        roi_pooled = roi_pooled.reshape(num_samples, -1)
        roi_pooled = self.roi_fc(roi_pooled)
        roi_pooled = self.roi_dropout(roi_pooled)
        roi_pooled = torch.relu(roi_pooled)

        data = torch.cat((spp_pooled, roi_pooled), dim=-1)
        data = self.fc7(data)
        data = self.fc7_dropout(data)
        data = torch.relu(data)

        data_mask = nn.Transformer.generate_square_subsequent_mask(num_samples)

        # data = data.reshape(B, S, -1)
        # data = self.model(data, data, src_mask = data_mask, tgt_mask = data_mask, src_is_causal = True, tgt_is_causal = True)
        # data = data.reshape(num_samples, -1)
        # data = self.fc8(data)
        # data = torch.sigmoid(data)

        data = data.reshape(B, S, -1)
        data, attn_weights = self.multihead_attn(data, data, data, attn_mask = data_mask, is_causal = True)
        data = data.reshape(num_samples, -1)
        data = self.fc8(data)
        data = torch.sigmoid(data)

        return data.reshape(B, S)

    def embed(self, frames, boxes=None):
        # B = batch size, C = channels, H = Height, W = width
        B, C, H, W = frames.shape

        if boxes is None:
            boxes = torch.FloatTensor([0, 0, 224, 224])
            boxes = boxes.repeat(B, 1)
            boxes = boxes.to(self.device)

        frames = self.backbone(frames)
        spp_pooled = self.spp(frames)
        spp_pooled = self.spp_fc(spp_pooled)
        spp_pooled = self.spp_dropout(spp_pooled)
        spp_pooled = torch.relu(spp_pooled)

        indices = torch.arange(0, B, device=self.device).reshape(B, -1)
        boxes = torch.cat((indices, boxes), dim=-1)

        roi_pooled = roi_align(
            frames, boxes, self.roi_size, spatial_scale=0.03125)
        roi_pooled = roi_pooled.reshape(B, -1)
        roi_pooled = self.roi_fc(roi_pooled)
        roi_pooled = self.roi_dropout(roi_pooled)
        roi_pooled = torch.relu(roi_pooled)

        return torch.cat((spp_pooled, roi_pooled), dim=-1)

    def reset(self):
        self.hidden1, self.hidden2 = None, None
