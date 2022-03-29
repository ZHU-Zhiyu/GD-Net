import torch
from SpaNet import SpaNet
from SpeNet import SpeNet,recon_net
import argparse
import torch.nn.functional as F
from torch.nn.init import xavier_normal_ as x_init
import itertools
from torchnet.logger import VisdomPlotLogger, VisdomLogger, VisdomTextLogger
import visdom
from torch.utils import data
from Hyper_loader_CAVE import Hyper_dataset
from torchvision.utils import make_grid
import numpy as np
from SSIM import SSIM
# from apex import amp
import os
import time
from torch.utils.data import DataLoader
import skimage.measure as skm
import skimage as ski
import random
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from lowrank import OLELoss_cuda_ralative,rank_constrain
from rgb2hsi import reconnet
import scipy.io as scio
from SpeNet import SpeNet
from thop import profile
from thop import clever_format
port = 8100
now = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
def xavier_init_(model):
    for name,param in model.named_parameters():
        if 'weight' in name:
            x_init(param)

def loss_spa(hsi,hsi_t,Mse):
    loss = Mse(hsi[1],hsi_t[1]) + Mse(hsi[2],hsi_t[2])
    return loss
def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()

def save_model(epoch,Renet):
    model_state_dict = {'spenet':Renet.state_dict()}
    os.makedirs('./save_model'+now,exist_ok=True)
    torch.save(model_state_dict,'./save_model'+now+'/state_dicr_{}.pkl'.format(epoch))

def load_model(model,mode_dict):
    # state_dict = torch.load(model_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in mode_dict.items():
        # name = k[7:] # remove `module.`
        name = k # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model
def get_spe_gt(hsi):
    output=[]
    output.append(hsi)
    index = [np.array(list(range(8)))*4,np.array(list(range(16)))*2]
    index[0][-1] = 30
    index[1][-1] = 30
    output.append(hsi[:,index[0],:,:])
    output.append(hsi[:,index[1],:,:])
    return output
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
conti = True
# grads = {}
# def save_grad(name):
#     def hook(grad):
#         grads[name] = grad
#     return hook
def psnr_cal(input,gt):
    mse = input - gt
    mse = mse**2
    mse = np.mean(np.array(mse))
    # mse = np.sqrt(mse)
    psnr_c = 10*np.log10(255**2/mse)
    return psnr_c

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--n_epochs', type = int, default = 10, help = 'number of epochs of training')
    parse.add_argument('--batch_size',type = int, default = 18, help = 'size of the batches')
    parse.add_argument('--lr',type = float, default = 1e-3, help='learing rate of network')
    parse.add_argument('--b1',type = float, default = 0.9, help='adam:decay of first order momentum of gradient')
    parse.add_argument('--b2',type = float, default = 0.999, help= 'adam: decay of first order momentum of gradient')
    parse.add_argument('--envname',type = str, default = 'temp', help= 'adam: decay of first order momentum of gradient')
    parse.add_argument('--local_rank', type=int, default=0)
    setup_seed(50)
    factor_lr = 1
    # spanet = SpaNet().cuda()
    opt = parse.parse_args()
    # torch.cuda.set_device(opt.local_rank)
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )
    Hyper_test = Hyper_dataset(output_shape=128,Training_mode='Test',data_name='CAVE')
    Hyper_test = DataLoader(Hyper_test,batch_size=1,shuffle=False, num_workers=4,pin_memory=True,
                                     drop_last=True)

    Hyper_train = Hyper_dataset(output_shape=128,Training_mode='Train',data_name='CAVE')
    datalen = Hyper_train.__len__()
    Hyper_train_sampler  = torch.utils.data.distributed.DistributedSampler(Hyper_train, shuffle=True)
    Hyper_train = DataLoader(Hyper_train,batch_size=opt.batch_size,shuffle=False, num_workers=10,pin_memory=True,
                                     drop_last=True,sampler=Hyper_train_sampler)
    device = torch.device('cuda:{}'.format(opt.local_rank))
    spenet = reconnet().cuda().to(device)
    opt = parse.parse_args()
    optimzier = torch.optim.Adam(itertools.chain(spenet.parameters()),lr = opt.lr,betas=(opt.b1,opt.b2),weight_decay=0)
    MaeLoss = torch.nn.L1Loss().to(device)   
    rank_constrain = rank_constrain().to(device)
    T_max = (datalen//(2*opt.batch_size))*500
    schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzier, T_max, eta_min=1e-5, last_epoch=-1) 
    env=opt.envname
    server='10.37.0.158'
    Loss_logger = VisdomPlotLogger('line',opts={'title':'loss2'},port=8100,env=env,server=server)
    Losst_logger = VisdomPlotLogger('line',opts={'title':'loss_t2'},port=8100,env=env,server=server)
    Lossm_logger = VisdomPlotLogger('line',opts={'title':'SSIM_t2'},port=8100,env=env,server=server)
    PSNR_logger = VisdomPlotLogger('line',opts={'title':'PSNR'},port=8100,env=env,server=server)
    train_logger = VisdomLogger('image',opts= {'title':'reconstruction image2'},port=8100,env=env,server=server)
    test_logger = VisdomLogger('image',opts={'title':'Residual image2'},port=8100,env=env,server=server)
    spenet = DDP(spenet, device_ids=[device], output_device=device,find_unused_parameters=True)
    b_r = 0
    ssim_best = 0
    psnr_best = 0
    if conti == True:
        state_dict = torch.load('/home/user_1/hyperspectral/spectralSR/cave9stages/save_model2020-08-30-13-19-58/state_dicr_499.pkl')
        spenet = load_model(spenet,state_dict['spenet'])
        del state_dict
    for epoch in range(opt.n_epochs):
        batch = 0
        loss_ = []
        loss_t = []
        ssim_log = []
        psnr_g = []
        if epoch %30 == 0:
            with torch.no_grad():
                for hsi,hsi_g,hsi_resize,msi in Hyper_test:
                    spenet.eval()
                    hsi_g = hsi_g.cuda().float()
                    batch += 1

                    msi = msi[-1].float().cuda()
                    # msi = msi[None,:,:,:]
                    print(msi.shape)
                    reconhsi,_ = spenet(msi)
                    fout = reconhsi
                    # hsi_g_ = (hsi_g*(2**8-1)).int().float().detach().cpu().numpy()
                    # fout_ = (fout*(2**8-1)).int().float().detach().cpu().numpy()
                    # flops, params = profile(spenet, inputs=msi)
                    # # hsi_spe,_,fout1 = spenet(hsi_resize_,msi_)
                    # flops, params = clever_format([flops, params], "%.3f")
                    # print('flops:{},params:{}'.format(flops,params))
                    # for i in range(31):
                    #     # psnr_g.append(skm.compare_psnr(hsi_g_[0,i,:,:],fout_[0,i,:,:],2**8-1))
                    #     psnr_g.append(psnr_cal(hsi_g_[0,i,:,:],fout_[0,i,:,:]))
                    # print(batch)
                    ssim = skm.compare_ssim(X =reconhsi[0,:,:,:].permute([1,2,0]).detach().cpu().float().numpy(), Y =hsi_g[0,:,:,:].permute([1,2,0]).detach().cpu().float().numpy(),K1 = 0.01, K2 = 0.03,multichannel=True)
                    ssim_log.append(ssim)
                    scio.savemat('/home/user_1/hyperspectral/spectralSR/cave9stages/'+str(batch)+'.mat',{'output':fout.detach().cpu().float().numpy(),'GT':hsi_g.detach().cpu().float().numpy()})
                    print('SSIM:{}'.format(np.mean(np.array(ssim_log))))
                    # scio.savemat('/home/zhu_19/data/eval/harvard_4_stage_rank_loss/'+str(batch)+'.mat',{'output':fout.detach().cpu().numpy(),'GT':hsi_g.detach().cpu().numpy()})
                # psnr_ = np.mean(np.array(psnr_g))
                # PSNR_logger.log(epoch,psnr_)