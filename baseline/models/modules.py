import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

# from util import log

import logging
log = logging.getLogger(__name__)


#https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight) 
        m.bias.data.fill_(0.01)     

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.double_conv.apply(init_weights)

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, stride=1)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down_s(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stride_conv = nn.Sequential(
            # nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, stride=2)
        )

    def forward(self, x):
        return self.stride_conv(x)

class Enc_Conv_splitCH(nn.Module):
    def __init__(self):
        super().__init__()

        #self.inc = DoubleConv(3, 64) # 64x224x224
        self.inch1 = DoubleConv(1, 64) # 64x224x224
        self.inch2 = DoubleConv(1, 64) # 64x224x224
        self.inch3 = DoubleConv(1, 64) # 64x224x224
        self.down1 = Down(64, 64) # 64x128x128
        self.down1s = DoubleConv(64, 64) # 64x112x112
        self.down2 = Down(64, 128) # 128x64x64
        self.down2s = DoubleConv(128, 128) # 128x56x56
        self.down3 = Down(128, 128) # 128x32x32
        self.down3s = DoubleConv(128, 128) # 128x28x28
        self.down4 = Down(128, 128) # 128x14x14 
        self.down4s = DoubleConv(128, 256) # 128x14x14
        self.down5 = Down(256, 256) # 128x7x7
        self.down5s = DoubleConv(256, 256) # 128x7x7 
        # self.down6 = Down(256, 256) # 128x3x3
        self.relu = nn.ReLU()

        self.conv_kv = nn.Conv2d(256, 128, 1, stride=1, padding=0)

    def forward(self, x):
        #Split Channels
        input_x = x
        x_ch1 = input_x[:, 0, :, :]
        x_ch2 = input_x[:, 1, :, :]
        x_ch3 = input_x[:, 2, :, :]
        x_ch1= x_ch1.unsqueeze(1)
        x_ch2= x_ch2.unsqueeze(1)
        x_ch3= x_ch3.unsqueeze(1)

        #First Channel
        x1 = self.inch1(x_ch1)  # [ B, 64, 224, 224]    
        x1 = self.down1s(self.down1(x1)) # [ B, 64, 112, 112]
        x1 = self.down2s(self.down2(x1))  # [ B, 128, 56, 56]
        x1 = self.down3s(self.down3(x1))  # [ B, 128, 28, 28]
        x1 = self.down4s(self.down4(x1)) # [ B, 256, 14, 14]
        conv_out_1 = self.down5s(self.down5(x1)) # [ B, 256, 7, 7]
        z_kv_1 = self.relu(self.conv_kv(conv_out_1))        # BxCxHxW 'relu' required   ---> [B, 128, 7, 7]
        z_kv_1 = torch.transpose(z_kv_1.view(input_x.shape[0], 128, -1), 2, 1) # B, C, HW   --->  [B, 49, 128]

        #Second Channel
        x2 = self.inch1(x_ch2)  # [ B, 64, 224, 224]    
        x2 = self.down1s(self.down1(x2)) # [ B, 64, 112, 112]
        x2 = self.down2s(self.down2(x2))  # [ B, 128, 56, 56]
        x2 = self.down3s(self.down3(x2))  # [ B, 128, 28, 28]
        x2 = self.down4s(self.down4(x2)) # [ B, 256, 14, 14]
        conv_out_2 = self.down5s(self.down5(x2)) # [ B, 256, 7, 7]
        z_kv_2 = self.relu(self.conv_kv(conv_out_2))        # BxCxHxW 'relu' required   ---> [B, 128, 7, 7]
        z_kv_2 = torch.transpose(z_kv_2.view(input_x.shape[0], 128, -1), 2, 1) # B, C, HW   --->  [B, 49, 128]

        #Third Channel
        x3 = self.inch1(x_ch3)  # [ B, 64, 224, 224]    
        x3 = self.down1s(self.down1(x3)) # [ B, 64, 112, 112]
        x3 = self.down2s(self.down2(x3))  # [ B, 128, 56, 56]
        x3 = self.down3s(self.down3(x3))  # [ B, 128, 28, 28]
        x3 = self.down4s(self.down4(x3)) # [ B, 256, 14, 14]
        conv_out_3 = self.down5s(self.down5(x3)) # [ B, 256, 7, 7]
        z_kv_3 = self.relu(self.conv_kv(conv_out_3))        # BxCxHxW 'relu' required   ---> [B, 128, 7, 7]
        z_kv_3 = torch.transpose(z_kv_3.view(input_x.shape[0], 128, -1), 2, 1) # B, C, HW   --->  [B, 49, 128]

        #Concat all Z
        all_z_kv = torch.cat( (z_kv_1, z_kv_2, z_kv_3), 2) # [ B, 49, 384]

        return all_z_kv 

class Enc_Conv_Psych(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_size = 128

        num_unit_1 = int(self.z_size / 4) #16
        num_unit_2 = int(self.z_size / 4) #32
        num_unit_3 = int(self.z_size / 2) #64
        num_unit_4 = int(self.z_size) #128

        self.num_unit_1 = num_unit_1
        self.num_unit_2 = num_unit_2
        self.num_unit_3 = num_unit_3
        self.num_unit_4 = num_unit_4

        self.relu = nn.ReLU()

        self.inch1 = DoubleConv(1, num_unit_1) # 64x224x224
        self.inch2 = DoubleConv(1, num_unit_1) # 64x224x224
        self.inch3 = DoubleConv(1, num_unit_2) # 64x224x224

        #Encoder for target and reference object
        self.down1_O = Down(num_unit_1, num_unit_1) # 128x128
        self.down1s_O = DoubleConv(num_unit_1, num_unit_1) # 112x112
        self.down2_O = Down(num_unit_1, num_unit_2) # 128x64x64
        self.down2s_O = DoubleConv(num_unit_2, num_unit_2) # 128x56x56
        self.down3_O = Down(num_unit_2, num_unit_2) # 128x32x32
        self.down3s_O = DoubleConv(num_unit_2, num_unit_2) # 128x28x28
        self.down4_O = Down(num_unit_2, num_unit_2) # 128x14x14 
        self.down4s_O = DoubleConv(num_unit_2, num_unit_3) # 128x14x14
        self.down5_O = Down(num_unit_3, num_unit_3) # 128x7x7
        self.down5s_O = DoubleConv(num_unit_3, num_unit_3) # 128x7x7 
        self.conv_kv_O = nn.Conv2d(num_unit_3, num_unit_2, 1, stride=1, padding=0)

        #Encoder for full images
        self.down1_F = Down(num_unit_2, num_unit_2) # 64x128x128
        self.down1s_F = DoubleConv(num_unit_2, num_unit_2) # 64x112x112
        self.down2_F = Down(num_unit_2, num_unit_3) # 128x64x64
        self.down2s_F = DoubleConv(num_unit_3, num_unit_3) # 128x56x56
        self.down3_F = Down(num_unit_3, num_unit_3) # 128x32x32
        self.down3s_F = DoubleConv(num_unit_3, num_unit_3) # 128x28x28
        self.down4_F = Down(num_unit_3, num_unit_3) # 128x14x14 
        self.down4s_F = DoubleConv(num_unit_3, num_unit_4) # 128x14x14
        self.down5_F = Down(num_unit_4, num_unit_4) # 128x7x7
        self.down5s_F = DoubleConv(num_unit_4, num_unit_4) # 128x7x7 

        self.conv_kv_F = nn.Conv2d(num_unit_4, num_unit_3, 1, stride=1, padding=0)


    def forward(self, input_img):
        #------------------------------
        # Split input to 3 channels
        #------------------------------
        x_ch1 = input_img[:, 0, :, :] #Target
        x_ch2 = input_img[:, 1, :, :] #Reference
        x_ch3 = input_img[:, 2, :, :] #Full Images

        x_ch1= x_ch1.unsqueeze(1)
        x_ch2= x_ch2.unsqueeze(1)
        x_ch3= x_ch3.unsqueeze(1)

        #------------------------------
        #Channel#1 = Target Objects
        #------------------------------
        x1 = self.inch1(x_ch1) # [ B, 64, 224, 224]
        x1 = self.down1s_O(self.down1_O(x1))  # [ B, 16, 112, 112]
        x1 = self.down2s_O(self.down2_O(x1))  # [ B, 32, 56, 56]
        x1 = self.down3s_O(self.down3_O(x1))  # [ B, 32, 28, 28]
        x1 = self.down4s_O(self.down4_O(x1)) # [ B, 64, 14, 14]
        conv_out_1 = self.down5s_O(self.down5_O(x1)) # [ B, 128, 7, 7]
        z_kv_1 = self.relu(self.conv_kv_O(conv_out_1))      # BxCxHxW 'relu' required   ---> [B, 32, 7, 7]
        z_kv_1 = torch.transpose(z_kv_1.view(input_img.shape[0], self.num_unit_2, -1), 2, 1) # B, C, HW   --->  [B, 49, 32]

        #------------------------------
        #Channel#2 = Reference Objects
        #------------------------------
        x2 = self.inch2(x_ch2) # [ B, 64, 224, 224]
        x2 = self.down1s_O(self.down1_O(x2))  # [ B, 16, 112, 112]
        x2 = self.down2s_O(self.down2_O(x2))  # [ B, 32, 56, 56]
        x2 = self.down3s_O(self.down3_O(x2))  # [ B, 32, 28, 28]
        x2 = self.down4s_O(self.down4_O(x2)) # [ B, 64, 14, 14]
        conv_out_2 = self.down5s_O(self.down5_O(x2)) # [ B, 64, 7, 7]
        z_kv_2 = self.relu(self.conv_kv_O(conv_out_2))      # BxCxHxW 'relu' required   ---> [B, 32, 7, 7]
        z_kv_2 = torch.transpose(z_kv_2.view(input_img.shape[0], self.num_unit_2, -1), 2, 1) # B, C, HW   --->  [B, 49, 32]

        #------------------------------
        #Channel#3 = Full Image 
        #------------------------------
        x3 = self.inch3(x_ch3) # [ B, 64, 224, 224]
        x3 = self.down1s_F(self.down1_F(x3))  # [ B, 32, 112, 112]
        x3 = self.down2s_F(self.down2_F(x3))  # [ B, 64, 56, 56]
        x3 = self.down3s_F(self.down3_F(x3))  # [ B, 64, 28, 28]
        x3 = self.down4s_F(self.down4_F(x3)) # [ B, 128, 14, 14]
        conv_out_3 = self.down5s_F(self.down5_F(x3)) # [ B, 128, 7, 7]
        z_kv_3 = self.relu(self.conv_kv_F(conv_out_3))      # BxCxHxW 'relu' required   ---> [B, 64, 7, 7]
        z_kv_3 = torch.transpose(z_kv_3.view(input_img.shape[0], self.num_unit_3, -1), 2, 1) # B, C, HW   --->  [B, 49, 64]

        #------------------------------
        # Concat for output
        #------------------------------
        all_z_kv = torch.cat( (z_kv_1, z_kv_2, z_kv_3), 2) # [ B, 49, 128]

        return all_z_kv 


class Enc_Conv_v0_16(nn.Module):
    def __init__(self):
        super().__init__()

        self.inc = DoubleConv(3, 64) # 64x224x224
        self.down1 = Down(64, 64) # 64x128x128
        self.down1s = DoubleConv(64, 64) # 64x112x112
        self.down2 = Down(64, 128) # 128x64x64
        self.down2s = DoubleConv(128, 128) # 128x56x56
        self.down3 = Down(128, 128) # 128x32x32
        self.down3s = DoubleConv(128, 128) # 128x28x28
        self.down4 = Down(128, 128) # 128x14x14 
        self.down4s = DoubleConv(128, 256) # 128x14x14
        self.down5 = Down(256, 256) # 128x7x7
        self.down5s = DoubleConv(256, 256) # 128x7x7 
        # self.down6 = Down(256, 256) # 128x3x3
        self.relu = nn.ReLU()

        self.conv_kv = nn.Conv2d(256, 128, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.inc(x)
        
        x = self.down1s(self.down1(x))
        x = self.down2s(self.down2(x))
        x = self.down3s(self.down3(x))
        x = self.down4s(self.down4(x))
        conv_out = self.down5s(self.down5(x))
        z_kv = self.relu(self.conv_kv(conv_out))        # BxCxHxW 'relu' required 
        z_kv = torch.transpose(z_kv.view(x.shape[0], 128, -1), 2, 1) # B, C, HW

        return z_kv 


class Resnet_block(nn.Module):
    def __init__(self, pretrained = False):
        super().__init__()
        
        self.value_size = 128 #args.value_size
        self.key_size = 128 #args.key_size
        # Inputs to hidden layer linear transformation
        self.resnet = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-2])
        self.conv_kv = nn.Conv2d(2048, self.value_size+self.key_size, 1, stride=1, padding=0)
        self.relu = nn.ReLU()

        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.resnet(x) # input = 1024
        z_kv = self.relu(self.conv_kv(x))       # BxCxHxW 'relu' required 
        z_kv = torch.transpose(z_kv.view(x.shape[0], self.value_size+self.key_size, -1), 2, 1) # B, C, HW
        z_keys, z_values = z_kv.split([self.key_size, self.value_size], dim=2)
        
        return z_keys, z_values 

class Resnet_block_128OUT(nn.Module):
    def __init__(self, pretrained = False):
        super().__init__()
        
        self.value_size = 64 #args.value_size
        self.key_size = 64 #args.key_size
        # Inputs to hidden layer linear transformation
        self.resnet = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-2])
        self.conv_kv = nn.Conv2d(2048, self.value_size+self.key_size, 1, stride=1, padding=0)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.resnet(x) # input = 1024
        z_kv = self.relu(self.conv_kv(x))       # BxCxHxW 'relu' required 
        z_kv = torch.transpose(z_kv.view(x.shape[0], self.value_size+self.key_size, -1), 2, 1) # B, C, HW
        #z_keys, z_values = z_kv.split([self.key_size, self.value_size], dim=2)
        #return z_keys, z_values 
        return z_kv     