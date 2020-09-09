#!/usr/bin/env python
# coding: utf-8

# # Imports 
# 

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import shutil, os
import glob
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import fastai
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *


# # Read Files from Test CSV

# In[2]:



def loadFileRleData():
    
    ### Data File for Image Ids
    path = Path('data/')
    
    id2labelcsv = pd.read_csv(path/'train-rle.csv')
    
    imageIds = id2labelcsv.ImageId
    rles = id2labelcsv.EncodedPixels

    id2rles = {}

    for i, id_key in tqdm(enumerate(imageIds)):
        value = rles[i]
        id2rles.setdefault(id_key, []).append(value)

    print('Master data file loaded with ids, paths, and rles...\n')
    print(imageIds.head(),'\n')
    print(rles.head())
    
    return id2labelcsv

    
    
loadFileRleData()


# # Import Data and Process IMGs

# In[3]:


import warnings
import sys
import glob

import scipy
from scipy.misc import imsave
from scipy import *
import scipy.ndimage as ndimage
from scipy.ndimage.filters import gaussian_filter
import gdcm
import cv2
import pydicom
from pydicom.data import get_testdata_files
from skimage import exposure
from skimage.filters import unsharp_mask


# In[4]:


def collectSumbissionData(sub_csv='sample_submission.csv', path = Path('data/')):

    #COLLECT TEST PATHS
    dcm_files = list(path.glob('**/*.dcm'))

    #MAP IDS TO PATH
    id2path = {}
    for f in dcm_files:
        f_path = Path(f)
        id2path.setdefault(f_path.stem, f)

    submission = pd.read_csv(path/sub_csv)
    imageIds = submission.ImageId
    
    ## OUTPUTS TO CONSOLE
    print(submission.head())
    print('\nTotal rows in file...  ',submission.size,'\n\n')
    print('looking up first submission id to confirm files in dir..\n\n')
    print(id2path[submission.ImageId[1]])
    
    return submission.ImageId, id2path
    
imageIds, id2path = collectSumbissionData();


# In[5]:


def processImages(imageIds, id2path, path=Path('data/'), display=False, testing=False):

    print('test data must be in /data... it will be saved in data/test-images \n')    
    
    dir = path/'test-images'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    tilesize=8
    cliplimit=2
    
    print('Processing images and saving them to... data/test-images')

    for i, id in tqdm(enumerate(imageIds)):
            
        if(testing==True and i > 100): break
            
        ## POST PROCESSING PIPLINE - TILESIZE:8, CLIP LIMIT:2 (Aug 25)
        img = pydicom.dcmread(str(id2path[id]))
        img = img.pixel_array

        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilesize,tilesize))
        img = clahe.apply(img)

        scipy.misc.imsave(dir/(id+'.dcm.png'), img)        


# In[6]:


#processImages(imageIds, id2path, display=False, testing=False)     


# # Combine Image Blocks 4 -> 1

# ### Block Utility Functions

# In[7]:


# https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays/16858283#
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


# In[8]:


def fourBlocks2image(nplist, blocksize):

    size = nplist[0].shape[-1]
    
    Image = np.zeros((2*size, 2*size), np.float32)

    Image[0:size, 0:size] = nplist[0]
    Image[0:size, size:2*size] = nplist[1]
    Image[size:2*size, 0:size] = nplist[2]
    Image[size:2*size, size:2*size] = nplist[3]
    
    return Image


# In[9]:


def image2blocks(img, parts_axis=2):
    
    assert img.data.shape[0] == 1, "expect image of shape [1,1024,1024]"

    img_data = img.data.reshape(img.data.shape[1:]).numpy()
    
    block_size = img_data.shape[0] // parts_axis

    img_blocks = blockshaped(img_data, block_size, block_size)
    
    for i, ima in enumerate(img_blocks):
        img = ima.astype(dtype=np.float32)
    
    return img_blocks


# ### Sanity Check of Blocks->Img & Imgs->Blocks

# In[10]:


def testImgToBlocks():

    path = Path('data/')
    
    img_f = path/'images'/'1.2.276.0.7230010.3.1.4.8323329.32752.1517875162.169303.dcm.png'
    img = open_image(img_f, convert_mode='L')
        
    img_blocks = image2blocks(img, parts_axis=2)

    fig, axs = plt.subplots(1, 4, figsize=(30,10))

    for i, ima in enumerate(img_blocks): 
        axs[i].imshow(ima, cmap='gray')
    
    return img_blocks
        


# In[11]:


sampleblocks = testImgToBlocks();
print(sampleblocks.shape)


# In[12]:


def testBlocksToImg(image_list=None):
    Image = fourBlocks2image(image_list, blocksize=512)
    plt.imshow(Image, cmap='gray')
    print(Image.shape)


# In[13]:


output = testBlocksToImg(image_list=sampleblocks)


# #  Inference

# In[14]:


def numpy2faiImg(img, size=None, path=Path('data/exported_models')):
    
    if not isinstance(img, np.ndarray):
        img = img.data.numpy()
    
    size = img.shape[-1]
    img = img.reshape(size,size)
    img = img.astype(dtype=np.float32)
    temp_path = path/('temp.png')
    scipy.misc.imsave(temp_path, img)
    faiImg = open_image(temp_path, convert_mode='L')
    
    return faiImg


# In[15]:


def matrix2Inference(img, learn= None, pkl='export.pkl', path=Path('data/exported_models')):

    if (learn==None):
        print('Loading model file: ',pkl )
        learn = load_learner(path, pkl)
        
    _,_,probs = learn.predict(img)
    
    return probs.squeeze()[1],learn


# In[16]:


def xray2Inference(xray, learn=None, pkl='export.pkl', path=Path('data/exported_models')):

    xray_size = xray.shape[2]

    ## Image >> Blocks
    img_blocks = image2blocks(xray, parts_axis=2)

    # Inference on each block
    Xs = []
    learn = None

    for i, block in enumerate(img_blocks):
        block = pil2tensor(block, dtype=np.float32)
        block = Image(block)

        faiImg = numpy2faiImg(block, size=img_blocks.shape[-1])
        # matrix2Inference --> (Probs, Learn)

        inference, learn = matrix2Inference(faiImg, learn, pkl, path)
        Xs.append(inference)

    prob_blocks = [prob.numpy() for prob in Xs]

    # COMBINE RESULTS
    xray_inference = fourBlocks2image(prob_blocks, blocksize=512)

    return xray_inference  # can add return learn if needed


# # Batch Inference

# In[17]:


def mask2rleFixed(img, width, height):
        rle = []
        lastColor = 0;
        currentPixel = 0;
        runStart = -1;
        runLength = 0;

        for x in range(width):
            for y in range(height):
                currentColor = img[x][y]
                if currentColor != lastColor:
                    if currentColor == 255:
                        runStart = currentPixel;
                        runLength = 1;
                    else:
                        rle.append(str(runStart));
                        rle.append(str(runLength));
                        runStart = -1;
                        runLength = 0;
                        currentPixel = 0;
                elif runStart > -1:
                    runLength += 1
                lastColor = currentColor;
                currentPixel += 1;
                if lastColor == 255:
                    rle.append(runStart)
                    rle.append(runLength)

        return " ".join(rle)


# ### Running Inference

# In[18]:


from matplotlib import rcParams

def plot(img_A, img_B, vmin=0.2, vmax=1):
    
    img_B = img_B.data.reshape(img.data.shape[1:]).numpy()
    img_B = img_B.astype(dtype=np.float32)

    # figure size in inches optional
    rcParams['figure.figsize'] = 22 ,16
    
    # display images
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img_A, vmin=vmin,vmax=vmax);
    ax[1].imshow(img_B, cmap="gray");


# In[19]:


import cv2
import time
import pandas as pd

from mask_functions import MaskRleCode as mr

path_img = Path('data/test-images')
fnames = get_image_files(path_img)

learn = none


# In[ ]:





# In[44]:



pkl1='export-v1.pkl'
pk12='export-v2.pkl'
pk12='export-v3.pkl'

path=Path('data/exported_models')



# In[36]:


learn = load_learner(path, pkl1)


# In[ ]:


id2rle = {}

for i, f in enumerate(fnames):
    



    TARGET_THR = 0.4

    img_path = fnames[i]

    img = open_image(img_path, convert_mode='L')

    inference = xray2Inference(img,learn=learn) 

    
    
    #inference,_ = matrix2Inference(img)

    #plot(inference, img, vmin=TARGET_THR, vmax=2)
    

    
    
    _ , mask = cv2.threshold(inference, TARGET_THR, 255, cv2.THRESH_BINARY)
    
    #plot(mask, img, vmin=TARGET_THR, vmax=2)
    
    mask = cv2.resize(mask, (1028,1028))
    mask = mask.astype(np.uint8)
    rle = mr.mask2rle(mask, 1028, 1028)
 
    if inference[inference>0.5].sum() < 15000:
        id2rle.setdefault(f.name[:-8], " -1")
    else :
        id2rle.setdefault(f.name[:-8], rle)

        
        
    print ("PARAMETERS FOR FILTERING...", inference.sum(),
           inference[inference>0.5].sum(), mask.sum())

    






# #     plot(inference, img,   vmin=TARGET_THR, vmax=1)
# #     plot(inference2, img,   vmin=TARGET_THR, vmax=1)



# In[ ]:





# In[47]:


df = pd.DataFrame(columns= ['ImageId', 'EncodedPixels'])

for Id, Rle in id2rle.items():
    df = df.append({'ImageId': Id, 'EncodedPixels': Rle}, ignore_index=True)

df.to_csv(index=False)


# In[73]:


path


# In[ ]:




