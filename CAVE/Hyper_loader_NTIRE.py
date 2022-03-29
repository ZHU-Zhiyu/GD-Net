import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import scipy.ndimage as scin
from scipy import ndimage
from get_name import get_name
import scipy.io as scio
import h5py
import lmdb
import os
import random
new_load =  lambda *a,**k: np.load(*a, allow_pickle=True, **k)
class Hyper_dataset(Dataset):
    """
    
    get the Hyperspectral image and corrssponding RGB image  
    use all data : high resolution HSI, high resolution MSI, low resolution HSI
    """
    def __init__(self, output_shape=512,ratio = 1,Training_mode='Train',data_name = 'CAVE',use_generated_data = False, use_all_data = True):
        # self.path = '/home/zhu_19/Hyperspectral_image/Hyperspectral_image_comparing_method/MHF-net-master/CAVEdata/'
        self.path = '/public/SSD/NTIRE2020_Train_Spectral/ARAD_HS_%04d.mat'
        self.path_2 = '/public/SSD/NTIRE2020_Train_Clean/ARAD_HS_%04d_clean.png'
        self.test_path = '/public/SSD/NTIRE2020_Validation_Clean/ARAD_HS_%04d_clean.png'
        self.test_path2 = '/public/SSD/NTIRE2020_Validation_Spectral/ARAD_HS_%04d.mat'
        # file_name = os.walk(self.path)
        # file_name = [i for i in file_name]
        # self.file_name = file_name[0][2]
        # self.shuffle_index  = [2,31,25,6,27,15,19,14,12,28,26,29,8,13,22,7,24,30,10,23,18,17,21,3,9,4,20,5,16,32,11,1]
        self.num_pre_img = 4
        self.TM = Training_mode
        self.train_len = 450*4**2
        self.test_len = 10
    def __len__(self):
        if self.TM == 'Train':
            return self.train_len
        elif self.TM == 'Test':
            return self.test_len
    # def zoom_img(self,input_img,ratio_):
    #     return np.concatenate([ndimage.zoom(img,zoom = ratio_)[np.newaxis,:,:] for img in input_img],0)
    def zoom_img(self,input_img,ratio_):
        # return np.concatenate([ndimage.zoom(img,zoom = ratio_)[np.newaxis,:,:] for img in input_img],0)
        output_shape = int(input_img.shape[-1]*ratio_)
        return np.concatenate([self.zoom_img_(img,output_shape = output_shape)[np.newaxis,:,:] for img in input_img],0)
    def zoom_img_(self,input_img,output_shape):
        return input_img.reshape(input_img.shape[0],output_shape,-1).mean(-1).swapaxes(0,1).reshape(output_shape,output_shape,-1).mean(-1).swapaxes(0,1)
    def recon_img(self, input_img):
        return cv2.resize(cv2.resize(input_img.transpose(1,2,0),dsize=(self.shape1,self.shape1)),dsize = (self.output_shape , self.output_shape)).transpose(2,0,1)
    def __getitem__(self, index):
        if self.TM == 'Test':
            index = index + self.train_len//(self.num_pre_img**2)+1
        if self.TM=='Train':
            # if self.direct_data == True:
            index_img = index // self.num_pre_img**2 +1
            # index_img = self.shuffle_index[index_img]-1
            index_inside_image = index % self.num_pre_img**2 
            index_row = index_inside_image // self.num_pre_img 
            index_col = index_inside_image % self.num_pre_img
            # hsi_g = scio.loadmat(self.path+'X/'+self.file_name[index_img])
            # msi = scio.loadmat(self.path+'Y/'+self.file_name[index_img])
            # hsi = scio.loadmat(self.path+'Z/'+self.file_name[index_img])
            hsi_g = scio.loadmat(self.path%index_img)
            msi = cv2.imread(self.path_2%index_img)

            # hsi_g = hsi_g['msi'][index_row*128:(index_row+1)*128,index_col*128:(index_col+1)*128,:]
            # hsi = hsi['Zmsi'][index_row*4:(index_row+1)*4,index_col*4:(index_col+1)*4,:]
            # msi = msi['RGB'][index_row*128:(index_row+1)*128,index_col*128:(index_col+1)*128,:]
            # factor = hsi_g['norm_factor']
            hsi_g = hsi_g['cube'][index_row*128- index_row*10:(index_row+1)*128 - index_row*10,index_col*128:(index_col+1)*128,:]
            # hsi = hsi['Zmsi'][index_row*4:(index_row+1)*4,index_col*4:(index_col+1)*4,:]
            msi = msi[index_row*128 - index_row*10:(index_row+1)*128 - index_row*10,index_col*128:(index_col+1)*128,:]
            
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            
            # Random rotation
            for j in range(rotTimes):
                hsi_g = np.rot90(hsi_g)
                # hsi = np.rot90(hsi)
                msi = np.rot90(msi)

            # Random vertical Flip   
            for j in range(vFlip):
                hsi_g = np.flip(hsi_g,axis=1)
                # hsi = np.flip(hsi,axis=1)
                msi = np.flip(msi,axis=1)
                # hsi_g = hsi_g[:,::-1,:]
                # hsi = hsi[:,::-1,:]
                # msi = msi[:,::-1,:]
        
            # Random Horizontal Flip
            for j in range(hFlip):
                hsi_g = np.flip(hsi_g,axis=0)
                # hsi = np.flip(hsi,axis=0)
                msi = np.flip(msi,axis=0)
                # hsi_g = hsi_g[::-1,:,:]
                # hsi = hsi[::-1,:,:]
                # msi = msi[::-1,:,:]
            # hsi = np.transpose(hsi,(2,0,1)).copy()
            msi = np.transpose(msi,(2,0,1)).copy() / 255.0
            hsi_g = np.transpose(hsi_g,(2,0,1)).copy()
            # print('shape of tensor {} {} {}'.format(hsi.shape,msi.shape,hsi_g.shape))
        elif self.TM=='Test':
            hsi_g = scio.loadmat(self.test_path2%index)
            msi = cv2.imread(self.test_path%index)
            # factor = hsi_g['norm_factor']
            # hsi = scio.loadmat(self.path+'Z/'+self.file_name[index])

            # hsi_g = hsi_g[index_row*128:(index_row+1)*128,index_col*128:(index_col+1)*128,:]
            hsi_g = np.transpose(hsi_g['cube'],(2,0,1))
            # hsi = hsi[index_row*4:(index_row+1)*4,index_col*4:(index_col+1)*4,:]
            # hsi = np.transpose(hsi['Zmsi'],(2,0,1))
            # msi = msi[index_row*128:(index_row+1)*128,index_col*128:(index_col+1)*128,:]
            msi = np.transpose(msi,(2,0,1)) / 255.0

        # if self.TM == 'Train':
        #     a = np.random.randint(low = 0,high= 3,size=1)
        #     # mean = np.mean(msi)
        #     if a <1:
        #         hsi_g = 1.0 - hsi_g
        #         msi = 1.0 - msi
        # hsi = self.zoom_img(hsi_g,1/32)
        # hsi_resize = hsi
        # hsi = self.zoom_img(hsi_g,1/32)
        # hsi_8 = self.zoom_img(hsi_g, 1/8)
        # hsi_2 = self.zoom_img(hsi_g, 1/2)
        # msi_8 = self.zoom_img(msi,1/8)
        # msi_2 = self.zoom_img(msi,1/2)
        hsi = hsi_g
        hsi_8 = hsi_g
        hsi_2 = hsi_g
        msi_8 = msi
        msi_2 = msi
        return ((hsi,hsi_8,hsi_2), hsi_g, hsi, (msi_8,msi_2,msi))
        