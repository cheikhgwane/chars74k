#!/usr/bin/env python
# coding: utf-8

# In[85]:


from keras.preprocessing import image
import tarfile
import os 
import platform
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base_dir = os.getcwd()

fnt_base_dir = os.path.join(base_dir,"EnglishFnt")
img_base_dir = os.path.join(base_dir,"EnglishImg")
hnd_base_dir = os.path.join(base_dir,"EnglishHnd")

#utils functions used to plot random image in dataset
def showFolderImageSample(base_folder,img_number=2):
    '''
    base_folder(String) : directory in which we'll look for sample
    img_number : for each folder how many img to show
    '''
    nrows = ncols = img_number
    
    fig = plt.gcf()
    fig.set_size_inches(ncols * img_number , nrows * img_number)
    dir_names =os.listdir(base_folder)
    
    #get sample directory
    sample_dir=[]
    for i in range(img_number):
        sample_dir.append(dir_names[randint(0,61)])
    dir_names = [os.path.join(base_folder,dname) for dname in sample_dir]
    
    print("total sample directory : {}".format(len(dir_names)))
    #for each sample directory get img_number random image
    
    img =[]
    for d in dir_names :
        for i in range(img_number):
            img.append(os.path.join(d,os.listdir(d)[randint(0,len(os.listdir(d)))]))
    
    for i,img_path in enumerate(img):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off') # Don't show axes (or gridlines)
            
        img = mpimg.imread(img_path)
        plt.imshow(img)
            
    plt.show()
    print("Size of one random image : {}".format(img.shape))

########################################################################
def datasetInfo(base_dir):
    print("Total training data  {}".format(len(os.listdir(os.path.join(base_dir,"training")))))
    print("Total validation data  {}".format(len(os.listdir(os.path.join(base_dir,"validation")))))
    print("Total test data  {}".format(len(os.listdir(os.path.join(base_dir,"test")))))

########################################################################
datasetInfo(fnt_base_dir)
showFolderImageSample(fnt_base_dir)


# In[91]:


# base model for font image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
#scale image to have it pixel values between 0 and 1
fnt_train_data = ImageDataGenerator(rescale=1./255)
fnt_validation_date = ImageDataGenerator(rescale=1./255)
fnt_test_data = ImageDataGenerator(rescale=1./255)


# In[ ]:




