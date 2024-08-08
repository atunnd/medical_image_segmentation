import torch
import torchvision.transforms.functional
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # First 3x3 convolutional layer
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        # Second 3x3 convolutional Layer
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        x = self.act2(x)
        
        return x

class DownSample(nn.Module):
    def __init__(self):
        super().__init__()

        # 2x2 pooling layer
        self.pool = nn.MaxPool2d(2) 
    
    def forward(self, x: torch.Tensor):
        return self.pool(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x:torch.Tensor):
        return self.up(x)

class CropAndConcat(nn.Module):
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        
        # crop the feature map from the contracting path to the size of the current feature map
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])

        # concatenate the feature maps
        x = torch.cat([x, contracting_x], dim=1)

        return x

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # contracting path (encoder)
        self.down_conv = nn.ModuleList([DoubleConv(i, o) for i, o in
                                        [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])

        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])
        
        # bottleneck
        self.middle_conv = DoubleConv(512, 1024)

        # expansive path (decoder)
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.up_conv = nn.ModuleList([DoubleConv(i, o) for i, o in
                                      [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):

        # collect feature maps for skip connection
        pass_through = []

        # contracting path
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x) # double convolution
            pass_through.append(x) # save feature map for skip connection
            x = self.down_sample[i](x) # max pooling 2d
        
        # 2 conv 3x3 at the bottom
        x = self.middle_conv(x)

        # expansive path
        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.concat[i](x, pass_through.pop())
            x = self.up_conv[i](x)
        
        x = self.final_conv(x)
        return x

