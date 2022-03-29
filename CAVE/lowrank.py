import torch
from torch.autograd import Variable, Function
import torch.nn as nn
import numpy as np
import scipy as sp
import scipy.linalg as linalg
import time
# import numba as nb
from torch_batch_svd import svd

# Inherit from Function
class OLELoss(Function):
    # def __init__(self, lambda_=0.25):
    #     super(OLELoss,self).__init__()
    #     self.lambda_ = lambda_
    #     self.ratio = 8 
    #     self.delta = 1e-2
    @staticmethod
    def forward(ctx, X, y):
        delta = 1e-2
        x = X.detach().cpu()
        # x shape B C H W
        y = y.detach().cpu()
        # y shape B C H W
        B,C,H,W = x.shape
        # patch size i
        i = 16
        with torch.no_grad():
            x = torch.reshape(x,[B,C,H//i,i,W//i,i])
            y = torch.reshape(y,[B,C,H//i,i,W//i,i])
            x = x.permute([2,4,0,1,3,5])
            y = y.permute([2,4,0,1,3,5])
            # H/8 , W/8 , B, C, 8, 8
            x = torch.reshape(x,[H//i*W//i*B,C,i*i])
            y = torch.reshape(y,[H//i*W//i*B,C,i*i])
            [Ux,Sx,Vx] = torch.svd(x)
            [Uy,Sy,Vy] = torch.svd(y)

            # sign matrix
            sign = torch.zeros_like(y).cpu()
            temp = Sx>Sy
            temp = temp.float()*2-1.0

            delta1 = Sx>delta
            delta2 = Sy>delta
            delta1 = delta1.float()
            delta2 = delta2.float()
            temp = temp*delta1*delta2
            temp = torch.diag_embed(temp)
            sign = temp

            obj = torch.abs(Sx - Sy)*delta1*delta2
            obj = torch.sum(obj,1)
            obj = torch.mean(obj)
            dx1 = torch.bmm(Ux,sign)
            dx1 = torch.bmm(dx1,Vx.permute([0,2,1]))
            dx2 = torch.bmm(Uy,sign)
            dx2 = torch.bmm(dx2,Vy.permute([0,2,1]))
            dX =  dx1 - dx2
            dX = torch.reshape(dX,[H//i,W//i,B,C,i,i])
            dX = dX.permute([2,3,0,4,1,5])
            dX = torch.reshape(dX,[B,C,H,W])
            ctx.save_for_backward(dX)
            # global dx3
            # dx3 = dX
        return torch.FloatTensor([float(obj)]).cuda()
    @staticmethod
    def backward(ctx, grad_output):

        [dX] = ctx.saved_variables
        return dX.cuda()*grad_output, None

class rank_constrain(nn.Module):
    def __init__(self,i= 48,j=48):
        super(rank_constrain,self).__init__()
        self.i = i
        self.j = j
    def forward(self,x,y):
        # patch size i
        i = self.i
        j = self.j
        patch_num=3
        stride = i//patch_num
        x = x.unfold(2,i,stride).unfold(3,j,stride)
        y = y.unfold(2,i,stride).unfold(3,j,stride)
        assert(x.shape == y.shape)
        B,C,p1,p2,s1,s2 = x.shape
        x = x.permute([0,2,3,1,4,5])
        y = y.permute([0,2,3,1,4,5])
        assert(s1 == i)
        assert(s2 == j)
        x = x.reshape([B*p1*p2,C,s1*s2])
        y = y.reshape([B*p1*p2,C,s1*s2])

        # H = H1//i*i
        # W = W1//j*j
        # x = x[:,:,:H,:W]
        # y = y[:,:,:H,:W]
        # x = torch.reshape(x,[B,C,H//i,i,W//j,j])
        # y = torch.reshape(y,[B,C,H//i,i,W//j,j])
        # x = x.permute([2,4,0,1,3,5])
        # y = y.permute([2,4,0,1,3,5])
        # x = torch.reshape(x,[H//i*W//j*B,C,i*j])
        # y = torch.reshape(y,[H//i*W//j*B,C,i*j])
        # x1 = x.permute([0,2,1])
        # x = torch.bmm(x,x1)
        # y1 = y.permute([0,2,1])
        # y = torch.bmm(y,y1)

        return OLELoss_cuda_patch.apply(x,y,i,j,patch_num*patch_num)


class rank_clip(Function):
    @staticmethod
    def forward(ctx, X,rank):
        # delta = 1e-2
        rank = rank.clamp(min = 0, max = 1)
        delta = 1e-3
        gamma = 1
        [B,C,H,W] = X.shape

        X = X.detach()
        X = X.reshape(B,C,H*W)
        # x shape B C H W
        rank = rank.detach()
        x1 = X.permute([0,2,1])
        x = torch.bmm(X,x1)

        B,C1,C2 = x.shape

        with torch.no_grad():
            [Ux,Sx,Vx] = svd(x)

            # Singular Value Decomposition
            Sx1 = Sx
            Sx = torch.sqrt(Sx)
            Sx1 = (Sx + 1e-6)**(-1)
            Sx1 = torch.diag_embed(Sx1)
            Vx = torch.bmm(Ux.permute([0,2,1]),X)
            Vx = torch.bmm(Sx1,Vx)

            # rank minimization
            rank_min = -rank + 1
            rank_min[:,0] = 1
            Sx = Sx * rank_min

            # matrix recover
            Sx = torch.diag_embed(Sx)
            output = torch.bmm(Sx,Vx)
            output = torch.bmm(Ux,output)
            # print(output.shape)
            output = output.reshape([B,C,H,W])
            
        # print('Rank loss average std:{},mean:{},max:{},min:{}.'.format(torch.mean(torch.std(dX,[0,2]),0),torch.mean(torch.abs(dX),[0,2]),torch.max(dX),torch.min(dX)))

        return output.cuda()
    @staticmethod
    def backward(ctx, grad_output):
        # [dX] = ctx.saved_variables
        # dX = dX.cuda()*grad_output
        # print('Rank loss average std:{},mean:{},max:{},min:{}.'.format(torch.mean(torch.std(dX,[2,3]),0),torch.mean(torch.abs(dX),[0,2,3]),torch.max(dX),torch.min(dX)))
        return grad_output, None


class rank_estimation_loss(Function):
    @staticmethod
    def forward(ctx, X, Y,rank):
        rank = rank.clamp(min = 0, max = 1)
        [B,C,H,W] = X.shape
        

        X = X.detach()
        # x shape B C H W
        Y = Y.detach()
        X = X.reshape(B,C,H*W)
        Y = Y.reshape(B,C,H*W)
        x1 = X.permute([0,2,1])
        x = torch.bmm(X,x1)
        y1 = Y.permute([0,2,1])
        y = torch.bmm(Y,y1)

        B,C1,C2 = x.shape

        with torch.no_grad():
            [Ux,Sx,Vx] = svd(x)
            [Uy,Sy,Vy] = svd(y)

            Sx1 = Sx
            Sy1 = Sy
            Sx = torch.sqrt(Sx)
            Sy = torch.sqrt(Sy)
            S2 = torch.stack([Sx,Sy],2)
            [S,_] = torch.max(S2,2)

            rank_GT = torch.abs(Sx - Sy)/(S + 1e-6)
            rank_GT[:,0] = 0
            print('GT')
            print(rank_GT)
            print('estimated')
            print(rank)
            obj = torch.mean(torch.abs(rank_GT - rank))
            rank_gradient = rank - rank_GT

            rank_gradient[:,0] = 0

            ctx.save_for_backward(rank_gradient)

        return torch.FloatTensor([float(obj)]).cuda()
    @staticmethod
    def backward(ctx, grad_output):
        
        [gradient] = ctx.saved_variables
        gradient = gradient.cuda()
        print('gradient')
        print(gradient)
        # print(gradient)
        return None, None, gradient


class OLELoss_cuda_patch(Function):
    @staticmethod
    def forward(ctx, X, Y,i,j,patch_num):
        # delta = 1e-2
        delta = 1e-3
        gamma = 1
        X = X.detach()
        # x shape B C H W
        Y = Y.detach()
        x1 = X.permute([0,2,1])
        x = torch.bmm(X,x1)
        y1 = Y.permute([0,2,1])
        y = torch.bmm(Y,y1)

        B,C1,C2 = x.shape

        with torch.no_grad():
            [Ux,Sx,Vx] = svd(x)
            [Uy,Sy,Vy] = svd(y)
            sign = torch.zeros_like(y)
            Sx1 = Sx
            Sy1 = Sy
            Sx = torch.sqrt(Sx)
            Sy = torch.sqrt(Sy)
            delta1 = Sx>delta
            delta2 = Sy>delta
            gamma1 = Sx<gamma
            delta1 = delta1.float()
            delta2 = delta2.float()
            gamma1 = gamma1.float()
            # S = torch.stack([Sx,Sy],2)
            S2 = torch.stack([Sx1,Sy1],2)
            [S,_] = torch.max(S2,2)
            S1 = torch.sqrt(S)
            temp = (Sx-Sy)/(S+1e-6)
            temp = delta1*delta2*gamma1
            temp = torch.diag_embed(temp)
            sign = temp

            dx1 = torch.bmm(Ux,sign)
            dx1 = torch.bmm(dx1,Ux.permute([0,2,1]))
            dX =  dx1
            # dX = dX/(2*C1*i*j*patch_num)
            dX = dX/(C1*i*j*patch_num)
            dX = torch.bmm(dX,X)
            ctx.save_for_backward(dX)
            
            # obj = torch.abs(Sx - Sy)/(S1+1e-6)*delta1*delta2*gamma1
            obj = delta1*delta2*gamma1
            obj = torch.sum(obj,1)/(C1*i*j)
            obj = torch.mean(obj)
        # print('Rank loss average std:{},mean:{},max:{},min:{}.'.format(torch.mean(torch.std(dX,[0,2]),0),torch.mean(torch.abs(dX),[0,2]),torch.max(dX),torch.min(dX)))

        return torch.FloatTensor([float(obj)]).cuda()
    @staticmethod
    def backward(ctx, grad_output):
        [dX] = ctx.saved_variables
        dX = dX.cuda()*grad_output
        # print('Rank loss average std:{},mean:{},max:{},min:{}.'.format(torch.mean(torch.std(dX,[2,3]),0),torch.mean(torch.abs(dX),[0,2,3]),torch.max(dX),torch.min(dX)))
        return dX, None,None,None,None

class fast_low_rank(Function):
    @staticmethod
    def forward(ctx, X, Y, rank):
        # delta = 1e-2
        delta = 1e-3
        gamma = 1
        X = X.detach()
        Y = Y.detach()
        B,C,H,W = X.shape
        X = X.reshape(B,C,H*W)
        Y = Y.reshape(B,C,H*W)
        # print(X.shape)

        x1 = X.permute([0,2,1])
        x = torch.bmm(X,x1)
        y1 = Y.permute([0,2,1])
        y = torch.bmm(Y,y1)

        B,C1,C2 = x.shape

        with torch.no_grad():
            [Ux,Sx,Vx] = svd(x)
            [Uy,Sy,Vy] = svd(y)
            Sy = torch.sqrt(Sy)
            
            Sx1 = Sx
            Sx = torch.sqrt(Sx)
            Sx11 = (Sx + 1e-4)**(-1)
            Sx1 = torch.diag_embed(Sx11)
            Sx = Sx *(1-rank)
            Sx = torch.diag_embed(Sx)
            Vx = torch.bmm(Ux.permute([0,2,1]),X)
            Vx = torch.bmm(Sx1,Vx)
            Xout = torch.bmm(Ux,Sx)
            Xout = torch.bmm(Xout,Vx)

            drank = (Sy - Sx11) / Sx11
            drank = drank < rank
            drank = drank.float() /(31)

        ctx.save_for_backward(drank)
        Xout = Xout.reshape(B,C,H,W)
        return Xout.float().cuda()
    @staticmethod
    def backward(ctx, grad_output):
        [dX] = ctx.saved_variables
        return grad_output,None, dX

class OLELoss_cpu(Function):
    @staticmethod
    def forward(ctx, X, y):
        delta = 1e-3
        x = X.detach()
        # x shape B C H W
        y = y.detach()
        # y shape B C H W
        B,C,H,W = x.shape
        # patch size i
        x = torch.reshape(x,[B,C,H*W])
        y = torch.reshape(y,[B,C,H*W])
        i = 16
        with torch.no_grad():
            [Ux,Sx,Vx] = torch.svd(x)
            [Uy,Sy,Vy] = torch.svd(y,compute_uv=False)
            S = torch.stack([Sx,Sy],2)
            [S,_] = torch.max(S,2)
            # sign matrix
            sign = torch.zeros_like(y).cpu()
            temp = Sx>Sy
            temp = temp.float()*2-1.0

            delta1 = Sx>delta
            delta2 = Sy>delta
            delta1 = delta1.float()
            delta2 = delta2.float()

            temp = (Sx - Sy)/(S+1e-6)*delta1*delta2
            temp = torch.diag_embed(temp)

            sign = temp

            obj = torch.abs(Sx - Sy)*delta1*delta2
            # obj = torch.sum(obj,1)
            obj = torch.mean(obj)
            dx1 = torch.bmm(Ux,sign)
            dx1 = torch.bmm(dx1,Vx.permute([0,2,1]))
            dx1 = torch.reshape(dx1,[B,C,H,W])
            # dx2 = torch.bmm(Uy,sign)
            # dx2 = torch.bmm(dx2,Vy.permute([0,2,1]))
            dX =  dx1
            # dX = torch.reshape(dX,[H//i,W//i,B,C,i,i])
            # dX = dX.permute([2,3,0,4,1,5])
            # dX = torch.reshape(dX,[B,C,H,W])
            ctx.save_for_backward(dX)
            # global dx3
            # dx3 = dX
        return torch.FloatTensor([float(obj)]).cuda()
    @staticmethod
    def backward(ctx, grad_output):
        # print self.dX
        [dX] = ctx.saved_variables
        return dX.cuda()*grad_output, None

class OLELoss_cuda(Function):
    @staticmethod
    def forward(ctx, X, y):
        delta = 1e-2
        x = X.detach()
        # x shape B C H W
        y = y.detach()
        # y shape B C H W
        B,C,H1,W1 = x.shape
        # patch size i
        i = 6
        j = 5
        H = H1//i*i
        W = W1//j*j
        x = x[:,:,:H,:W]
        y = y[:,:,:H,:W]

        with torch.no_grad():
            x = torch.reshape(x,[B,C,H//i,i,W//j,j])
            y = torch.reshape(y,[B,C,H//i,i,W//j,j])
            x = x.permute([2,4,0,1,3,5])
            y = y.permute([2,4,0,1,3,5])
            x = torch.reshape(x,[H//i*W//j*B,C,i*j])
            y = torch.reshape(y,[H//i*W//j*B,C,i*j])
            [Ux,Sx,Vx] = svd(x)
            [Uy,Sy,Vy] = svd(y)
            sign = torch.zeros_like(y)
            temp = Sx>Sy
            temp = temp.float()*2-1.0

            delta1 = Sx>delta
            delta2 = Sy>delta
            delta1 = delta1.float()
            delta2 = delta2.float()
            temp = temp*delta1*delta2
            temp = torch.diag_embed(temp)
            sign = temp

            obj = torch.abs(Sx - Sy)*delta1*delta2

            obj = torch.mean(obj)
            dx1 = torch.bmm(Ux,sign)

            dx1 = torch.bmm(dx1,Vx.permute([0,2,1]))
            dX =  dx1
            dX = torch.reshape(dX,[H//i,W//j,B,C,i,j])
            dX = dX.permute([2,3,0,4,1,5])
            dX = torch.reshape(dX,[B,C,H,W])
            a = torch.zeros((B,C,H,W1-W),device=dX.device)
            b = torch.zeros((B,C,H1-H,W1),device=dX.device)
            dX = torch.cat([dX,a],dim=3)
            dX = torch.cat([dX,b],dim=2)
            dX = dX/(H*W*C)
            ctx.save_for_backward(dX)
            # global dx3
            # dx3 = dX
        return torch.FloatTensor([float(obj)]).cuda()
    @staticmethod
    def backward(ctx, grad_output):
        [dX] = ctx.saved_variables
        dX = dX.cuda()*grad_output
        # print('Rank loss average std:{},mean:{},max:{},min:{}.'.format(torch.mean(torch.std(dX,[2,3]),0),torch.mean(torch.abs(dX),[0,2,3]),torch.max(dX),torch.min(dX)))
        return dX, None

class OLELoss_cuda_ralative(Function):
    @staticmethod
    def forward(ctx, X, y):
        # delta = 1e-2
        delta = 1e-3
        x = X.detach()
        # x shape B C H W
        y = y.detach()
        # y shape B C H W
        B,C,H1,W1 = x.shape
        # patch size i
        i = 6
        j = 5
        H = H1//i*i
        W = W1//j*j
        x = x[:,:,:H,:W]
        y = y[:,:,:H,:W]

        with torch.no_grad():
            x = torch.reshape(x,[B,C,H//i,i,W//j,j])
            y = torch.reshape(y,[B,C,H//i,i,W//j,j])
            x = x.permute([2,4,0,1,3,5])
            y = y.permute([2,4,0,1,3,5])
            x = torch.reshape(x,[H//i*W//j*B,C,i*j])
            y = torch.reshape(y,[H//i*W//j*B,C,i*j])
            [Ux,Sx,Vx] = svd(x)
            [Uy,Sy,Vy] = svd(y)
            sign = torch.zeros_like(y)
            temp = Sx>Sy
            temp = temp.float()*2-1.0

            delta1 = Sx>delta
            delta2 = Sy>delta
            delta1 = delta1.float()
            delta2 = delta2.float()
            S = torch.stack([Sx,Sy],2)
            [S,_] = torch.max(S,2)
            # temp = (Sx-Sy)/(S+1e-6)*torch.mean(S,1,keepdim=True)
            temp = (Sx-Sy)/(S+1e-6)
            temp = temp*delta1*delta2
            temp = torch.diag_embed(temp)
            sign = temp

            obj = torch.abs(Sx - Sy)*delta1*delta2

            obj = torch.mean(obj)
            dx1 = torch.bmm(Ux,sign)
            
            dx1 = torch.bmm(dx1,Vx.permute([0,2,1]))
            dX =  dx1
            dX = torch.reshape(dX,[H//i,W//j,B,C,i,j])
            dX = dX.permute([2,3,0,4,1,5])
            dX = torch.reshape(dX,[B,C,H,W])
            a = torch.zeros((B,C,H,W1-W),device=dX.device)
            b = torch.zeros((B,C,H1-H,W1),device=dX.device)
            dX = torch.cat([dX,a],dim=3)
            dX = torch.cat([dX,b],dim=2)
            dX = dX/(H*W*C)
            ctx.save_for_backward(dX)

        return torch.FloatTensor([float(obj)]).cuda()
    @staticmethod
    def backward(ctx, grad_output):
        [dX] = ctx.saved_variables
        dX = dX.cuda()*grad_output
        # print('Rank loss average std:{},mean:{},max:{},min:{}.'.format(torch.mean(torch.std(dX,[2,3]),0),torch.mean(torch.abs(dX),[0,2,3]),torch.max(dX),torch.min(dX)))
        return dX, None

class OLELoss_cuda_MODE(Function):
    # def __init__(self, lambda_=0.25):
    #     super(OLELoss,self).__init__()
    #     self.lambda_ = lambda_
    #     self.ratio = 8 
    #     self.delta = 1e-2
    @staticmethod
    def forward(ctx, X, y):
        delta = 1e-2
        x = X.detach()
        # x shape B C H W
        y = y.detach()
        # y shape B C H W
        B,C,H1,W1 = x.shape
        # patch size i
        i = 6
        j = 5
        H = H1//i*i
        W = W1//j*j
        x = x[:,:,:H,:W]
        y = y[:,:,:H,:W]
        with torch.no_grad():
            x = torch.reshape(x,[B,C,H//i,i,W//j,j])
            y = torch.reshape(y,[B,C,H//i,i,W//j,j])
            x = x.permute([2,4,0,1,3,5])
            y = y.permute([2,4,0,1,3,5])
            # H/8 , W/8 , B, C, 8, 8
            x = torch.reshape(x,[H//i*W//j*B,C,i*j])
            y = torch.reshape(y,[H//i*W//j*B,C,i*j])
            [Ux,Sx,Vx] = svd(x)
            [Uy,Sy,Vy] = svd(y)
            sign = torch.zeros_like(y)
            temp = Sx>Sy
            temp = temp.float()*2-1.0

            delta1 = Sx>delta
            delta2 = Sy>delta
            delta1 = delta1.float()
            delta2 = delta2.float()
            temp = temp*delta1*delta2
            temp = torch.diag_embed(temp)
            sign = temp

            obj = torch.abs(Sx - Sy)*delta1*delta2
            obj = torch.sum(obj,1)
            obj = torch.mean(obj)
            dx2 = torch.bmm(Uy,sign)
            dx2 = torch.bmm(dx2,Vy.permute([0,2,1]))
            dX =  -dx2
            dX = torch.reshape(dX,[H//i,W//j,B,C,i,j])
            dX = dX.permute([2,3,0,4,1,5])
            dX = torch.reshape(dX,[B,C,H,W])
            a = torch.zeros((B,C,H,W1-W),device=dX.device)
            b = torch.zeros((B,C,H1-H,W1),device=dX.device)
            dX = torch.cat([dX,a],dim=3)
            dX = torch.cat([dX,b],dim=2)
            ctx.save_for_backward(dX)
            
        return torch.FloatTensor([float(obj)]).cuda()
    @staticmethod
    def backward(ctx, grad_output):

        [dX] = ctx.saved_variables

        return dX.cuda()*grad_output, None

# @nb.njit(fastmath=True,parallel=True,nopython=True)
# def bmm_num(x,y):
#     B,C,W = x.shape
#     B,W,K = y.shape
#     output = np.zeros([B,C,K],dtype=np.float32)
#     for j in range(B):
#         output[j,:,:] = x[j,:,:]@y[j,:,:]
#     return output

# @nb.njit(fastmath=True,parallel=True,nopython=True)
# def SVDnum(x,y,H,W,B,C,i):
#     Ux = np.zeros((H//i*W//i*B,C,C),dtype=x.dtype)
#     Sx = np.zeros((H//i*W//i*B,C),dtype=x.dtype)
#     Vx = np.zeros((H//i*W//i*B,C,i*i),dtype=x.dtype)
#     Uy = np.zeros((H//i*W//i*B,C,C),dtype=x.dtype)
#     Sy = np.zeros((H//i*W//i*B,C),dtype=x.dtype)
#     Vy = np.zeros((H//i*W//i*B,C,i*i),dtype=x.dtype)
#     for j in range(x.shape[0]):
#         [Ux[j,:,:],Sx[j,:],Vx[j,:,:]] = np.linalg.svd(x[j,:,:],full_matrices=False)
#         # [Uy[j,:,:],Sy[j,:],Vy[j,:,:]] = np.linalg.svd(y[j,:,:],full_matrices=False,compute_uv=False)
#         Sy[j,:]= np.linalg.svd(y[j,:,:],full_matrices=False,compute_uv=False)
#     return Ux, Sx, Vx, Uy, Sy, Vy
#   for l in nb.prange(x.shape[0]):
#     for i in range(T.shape[0]):
#       sum=0.
#       for j in range(T.shape[1]):
#         for k in range(T.shape[2]):
#           sum+=T[i,j,k]*x[l,j]*x[l,k]
#       res[l,i]=sum
#   return res

class OLELossnum(Function):
    # def __init__(self, lambda_=0.25):
    #     super(OLELoss,self).__init__()
    #     self.lambda_ = lambda_
    #     self.ratio = 8 
    #     self.delta = 1e-2
    @staticmethod
    def forward(ctx, X, y):
        delta = 1e-2
        x = X.detach().cpu().numpy().astype(np.float32)
        # x shape B C H W
        y = y.detach().cpu().numpy().astype(np.float32)
        # y shape B C H W
        B,C,H,W = x.shape
        # patch size i
        i = 16
        x = np.reshape(x,[B,C,H//i,i,W//i,i])
        y = np.reshape(y,[B,C,H//i,i,W//i,i])
        x = np.transpose(x,[2,4,0,1,3,5])
        y = np.transpose(y,[2,4,0,1,3,5])
        # H/8 , W/8 , B, C, 8, 8
        x = np.reshape(x,[H//i*W//i*B,C,i*i])
        y = np.reshape(y,[H//i*W//i*B,C,i*i])
        # Ux, Sx1, Vx, Uy, Sy1, Vy = SVDnum(x,y,H,W,B,C,i)
        Ux, Sx1, Vx = np.linalg.svd(x,full_matrices=False)
        Sy1 = np.linalg.svd(y,full_matrices=False,compute_uv=False)
        index = np.arange(C)
        index1 = np.repeat(index[np.newaxis,:],H//i*W//i*B,axis=0)
        index_batch = np.repeat(index[:,np.newaxis],H//i*W//i*B,axis=1)
        index_batch = index_batch.reshape(-1)
        index1 = index1.reshape(-1)
        # Sx = np.zeros([H//i*W//i*B,C,C])
        # Sy = np.zeros([H//i*W//i*B,C,C])
        # a = np.diag_indices(C,H//i*W//i*B)
        # Sx[(index_batch,index1,index1)] = np.reshape(Sx1,[-1])
        # Sy[(index_batch,index1,index1)] = np.reshape(Sy1,[-1])

        temp = Sx1>Sy1
        temp = temp.astype(np.float32)*2.0-1.0

        delta1 = Sx1>delta
        delta2 = Sy1>delta
        delta1 = delta1.astype(np.float32)
        delta2 = delta2.astype(np.float32)
        # print('shape of temp:{}, delta1:{}, delta2:{}'.format(temp.shape, delta1.shape, delta2.shape))
        temp = temp*delta1*delta2
        # temp = torch.diag_embed(temp)
        sign = np.zeros([H//i*W//i*B,C,C])
        sign[(index_batch,index1,index1)] = np.reshape(temp,[-1])

        obj = np.abs(Sx1 - Sy1)*delta1*delta2
        obj = np.sum(obj,1)
        obj = np.mean(obj)
        dx1 = np.einsum('ijk,ikl->ijl',Ux,sign)
        # print('shape of Ux:{},sign:{},Vx:{},dx1:{}'.format(Ux.shape,sign.shape,Vx.shape,dx1.shape))
        dx1 = np.einsum('ijk,ikl->ijl',dx1,Vx)
        # dx2 = bmm_num(Uy,sign)
        # dx2 = bmm_num(dx2,Vy)
        dX = dx1
        # print('shape of dx:{}'.format(dX.shape))
        dX = np.reshape(dX,[H//i,W//i,B,C,i,i])
        dX = dX.permute([2,3,0,4,1,5])
        dX = np.transpose(dX,[B,C,H,W])
        dX = torch.from_numpy(dX)
        ctx.save_for_backward(dX)
        return torch.FloatTensor([float(obj)]).cuda()
    @staticmethod
    def backward(ctx, grad_output):
        # print self.dX
        [dX] = ctx.saved_variables
        return dX.cuda()*grad_output, None

# @nb.njit(fastmath=True,parallel=True,nopython=True)
# def SVDtemp(x,B,C,L):
#     Ux = np.zeros((B,C,C),dtype=x.dtype)
#     Sx = np.zeros((B,C),dtype=x.dtype)
#     Vx = np.zeros((B,C,L),dtype=x.dtype)
#     for j in range(x.shape[0]):
#         [Ux[j,:,:],Sx[j,:],Vx[j,:,:]] = np.linalg.svd(x[j,:,:],full_matrices=False)
#     return Ux, Sx, Vx


# if __name__=='__main__':
#     a = np.random.randn(10,31,128*128)
#     B,C,L = a.shape
#     t1 = time.time()
#     Ux, Sx, Vx = SVDtemp(a,B,C,L)
#     t2 = time.time()
#     U,S,V = np.linalg.svd(a)
#     print('numba used time:{},numpy used time:{}'.format(t2-t1,time.time() - t2))