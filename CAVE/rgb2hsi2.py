import torch
import torch.nn as nn
# from SpaNet import BCR, denselayer
import numpy as np
import torch.nn.functional as f
from lowrank import rank_clip
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
        
class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class My_Bn(nn.Module):
    def __init__(self):
        super(My_Bn,self).__init__()
    def forward(self,x):
        # print(x.shape)
        _,C,_,_ = x.shape
        x1,x2 = torch.split(x,C//2,dim=1)
        x1 = x1 - nn.AdaptiveAvgPool2d(1)(x1)
        x1 = torch.cat([x1,x2],dim=1)
        return x1


class BCR(nn.Module):
    def __init__(self,kernel,cin,cout,group=1,stride=1,RELU=True,padding = 0,BN=False):
        super(BCR,self).__init__()
        if stride > 0:
            self.conv = nn.Conv2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=stride,padding= padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=int(abs(stride)),padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.Swish = MemoryEfficientSwish()
        
        if RELU:
            if BN:
                self.Bn = My_Bn()
                self.Module = nn.Sequential(
                    self.Bn,
                    self.conv,
                    self.relu,
                )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                    self.relu
                )
        else:
            if BN:
                self.Bn = My_Bn()
                self.Module = nn.Sequential(
                    self.Bn,
                    self.conv,
                )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                )

    def forward(self, x):
        output = self.Module(x)
        return output

class denselayer(nn.Module):
    def __init__(self,cin,cout=31,RELU=True,BN=True,stride=1):
        super(denselayer, self).__init__()
        self.compressLayer = BCR(kernel=1,cin=cin,cout=cout,RELU=True,BN=BN)
        self.actlayer = BCR(kernel=3,cin=cout,stride=stride,cout=cout,group=cout,RELU=RELU,padding=1,BN=BN)
    def forward(self, x):
        output = self.compressLayer(x)
        output = self.actlayer(output)

        return output

class stage(nn.Module):
    def __init__(self,cin,cout,final=False,extra=0):
        super(stage,self).__init__()
        self.Upconv = BCR(kernel = 3,cin = cin, cout = 2*cout,stride= 1,padding=1,RELU=False,BN=False)
        if final == True:
            f_cout = cout +1
        else:
            f_cout = cout
        mid = cout*3
        self.denselayers = nn.ModuleList([
            denselayer(cin=2*cout,cout=cout*2),
            denselayer(cin=4*cout,cout=cout*2),
            denselayer(cin=6*cout,cout=cout*2),
            denselayer(cin=8*cout,cout=cout*2),
            denselayer(cin=10*cout,cout=f_cout,RELU=False,BN=False)])
    def forward(self,MSI,extra_data=None):
        MSI = self.Upconv(MSI)
        # print('-----------ERROR------------')
        # print(MSI.shape)
        x = [MSI]
        # print(MSI.shape)
        for layer in self.denselayers:
            x_ = layer(torch.cat(x,1))
            x.append(x_)
        
        return x[-1]

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class clamp(torch.nn.Module):
    def forward(self, x):
        x = x.clamp(min = 0, max = 1)
        return x

class reconnet(nn.Module):
    def __init__(self,extra=[0,0,0]):
        super(reconnet,self).__init__()

        self.stages = nn.ModuleList([
            stage(cin=3,cout=31,extra = extra[0]),
            stage(cin=3,cout=31,extra = extra[1]),
            stage(cin=3,cout=31,extra = extra[2])])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.degradation = BCR(kernel=3,cin = 31, cout= 3,group=1,stride=1,RELU=False,BN=False,padding=1)
        self.rankestimation = nn.Sequential(
            nn.Conv2d(in_channels=31, out_channels= 64,stride= 2,kernel_size= 3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels= 128,stride= 2,kernel_size= 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(in_features=128, out_features= 64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features= 31),
            # nn.Sigmoid()
            )
        # nn.Sequential(denselayer(cin = 31, cout= 64,RELU=True,BN=False,stride=2),
        #     denselayer(cin = 64, cout= 128,RELU=True,BN=False,stride=2),
        #     nn.AdaptiveAvgPool2d(1),
        #     Flatten(),
        #     nn.Linear(in_features=128, out_features= 64),
        #     nn.ReLU(),
        #     nn.Linear(in_features=64, out_features= 31),
        #     nn.Sigmoid()
        # )
        #  nn.Sequential(
        #     nn.Conv2d(in_channels=31, out_channels= 64,stride= 2,kernel_size= 3),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels= 128,stride= 2,kernel_size= 3),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d(1),
        #     Flatten(),
        #     nn.Linear(in_features=128, out_features= 64),
        #     nn.ReLU(),
        #     nn.Linear(in_features=64, out_features= 31),
        #     nn.Sigmoid())
        # #  nn.Sequential(
        #     nn.Conv2d(in_channels=31, out_channels= 64,stride= 2,kernel_size= 3),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels= 128,stride= 2,kernel_size= 3),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d(1),
        #     Flatten(),
        #     nn.Linear(in_features=128, out_features= 64),
        #     nn.ReLU(),
        #     nn.Linear(in_features=64, out_features= 31),
        #     nn.Sigmoid()
        # ),
        # ])
    def forward(self,MSI,extra_data=None):

        ref = [np.array(range(8))*4, np.array(range(16))*2]
        ref[0][-1] = 30
        ref[1][-1] = 30
        recon_out = None
        recon_list = []
        MSI = [MSI]
        rank_list = []
        for index , stage in enumerate(self.stages):
            recon = stage(MSI[-1])
            if recon_out is None:
                recon_out = recon
            else:
                recon_out = recon_out + recon
            recon_list.append(recon_out)
            rank_recude = self.rankestimation(recon_out.detach())
            recon_out   = rank_clip.apply(recon_out,rank_recude)
            rank_list.append(rank_recude)
            # output = torch.stack(recon_out,dim=0).sum(dim=0)
            msi_ = MSI[0] - self.degradation(recon_out)
            MSI.append(msi_)
        # [rank,_] = torch.sort(rank)
        # print(rank.shape)
        # rank = rank - rank[:,0][:,None]
        
        return recon_out, rank_list,recon_list