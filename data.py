# -*- coding: utf-8 -*-
"""Data

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DWH-UMQXB1S58fUQxmODVwBkd8gaYKZX
"""





'''for taking images path as an input and then converting those images into 
  noisy images for training purpose as model requires noisy images to train,after adding gaussian noise, other noises can also be input
  by using their specific functions.
'''
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from sys import platform
from random import choice
from string import ascii_letters
class NoiseData(Dataset):
    def __init__(self, directory, dimensionCrop=128, NoiseModel=('gaussain',50),CleanTargets=False):
        self.directory=directory  
        self.dimensionCrop=dimensionCrop
        self.CleanTargets=CleanTargets
        self.Noise= NoiseModel[0]
        self.NoiseParam=NoiseModel[1]
        self.images=os.listdir(directory)
 
    def croppingimages(self,images):
#method for cropping images at random postion for specific crop dimension i.e. dimensionCrop
      try:
        imgwidth=images[0].size
        imgheight=images[1].size
        if imgwidth>=self.dimensionCrop and imgheight>=self.dimensionCrop:
            result=[]
            i=np.random.randint(0,imgheight-self.dimensionCrop+2)
            j=np.random.randint(0,imgwidth-self.dimensionCrop+2)
            for probe in images:
                if min(imgwidth,imgheight)< self.dimensionCrop:
                    probe=tvF.resize(img,(self.dimensionCrop, self.dimensionCrop))
                result.append(tvF.crop(probe,i,j,self.dimensionCrop,self.dimensionCrop))
            return result
      except:
          print("unable to crop!")
 
    def Noise_text(self, image):
#method for adding text overlay as noise on the image 
          width=image.size[0]
          height=image.size[1]
          c=len(image.getbands())
          #default text overlay would be of Times New Roman
          serif='Times New Roman.ttf'
          textImage=image.copy()
          textDraw=imageDraw.Draw(textImage)
          maskImage=Image.new('1',(width,height))
          occupancyMax=np.random.uniform(0,self.NoiseParam)   
          def getOccupancy(x):
              y=np.array(x,np.uint8)
              return np.sum(y)/ y.size
          while True:
            current_font=ImageFont.truetype(serif, np.random.randint(10,21))
            length=np.random.randint(10,20)
            char =''.join(choice(ascii_letters) for i in range(length))
            color=tuple(np.random.randint(0,255,c))
            position=(np.random.randint(0,w),np.random.randint(0,h))
            textDraw.text(position,chars,color,font=current_font)
            if getOccupancy(maskImage)> occupancyMax:
              break
          return{'image':textImage,"mask":None,"use_mask":False}
    def Gaussian(self,image):
#method for adding gaussain noise to the image
        imgwidth=image.size[0]
        imgheight=image.size[1]
        color=len(image.getbands())
        stand=np.random.uniform(0,self.NoiseParam)
        noise=np.random.normal(0,stand,(imgwidth,imgheight,color))
        GaussainImage=np.array(image)+noise
        noisy_image=no.clip(GaussainImage,0,255).astype(np.uint8)
        return{"image":Image.fromarray(noisy_image),'mask':None,'use_mask':False}
    def Poisson(self,image):
#method for adding Poisson noise to the image
        noise_mask=np.random.poisson(np.array(image))
        return {'image':noise_mask.astype(np.unint8),'mask':None,'use_mask':False}
    def Bernoulli(self,image):
#method to add multiplicative bernoulli noise 
        siz=np.array(image).shape[0]
        probability=random.uniform(0,self.NoiseParam)
        Imagemask=np.random.choice([0,1], size=(siz,siz),p=[probability,1-probability])
        Imagemask=np.repeat(mask[:,:,np.newaxix],3,axis=2)
        return {'image':np.multiply(image,Imagemask).astype(np.unit8),'mask':mask.astype}              
 
 
    def ChooseCorruption(self, image): 
       #Method for choosing the required Corruption for the image
        if self.noise == 'gaussian':
            return self.Gaussian(image)
        elif self.noise == 'poisson':
            return self.Poisson(image)
        elif self.noise == 'multiplicative_bernoulli':
            return self.Bernoulli(image)
        else:
            raise ValueError('Not valid Corruption!')
 
    def __getitem__(self, index):
        #Corrupting the image then saving it
        ImagePath = os.path.join(self.directory, self.images[index])
        image = Image.open(ImagePath).convert('RGB')
 
        if self.crop_size > 0:
            image = self._random_crop_to_size([image])[0]
 
        DictionaryImages = self.ChooseCorruption(image)
        DictionaryImages['image'] = tvF.to_tensor(DictionaryImages['image'])
 
        if DictionaryImages['use_mask']:
            DictionaryImages['mask'] = tvF.to_tensor(DictionaryImages['mask'])
 
        if self.CleanTargets:
            #print('clean target')
            target = tvF.to_tensor(image)
        else:
            #print('corrupt target')
            DictionaryTarget = self.ChooseCorruption(image)
            target = tvF.to_tensor(DictionaryTarget['image'])
 
        if DictionaryImages['use_mask']:
            return [DictionaryImages['image'], DictionaryImages['mask'], target]
        else:
            return [DictionaryImages['image'], target]
 
    def __len__(self):
		   return len(self.images)