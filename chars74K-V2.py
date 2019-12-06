#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.preprocessing import LabelEncoder
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import layers
from keras import models
from random import randint
import math
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
# path to directory
base_dir = "C:\\Users\\Cheikh\\Desktop\\projetChars74k\\merged"

# utils functions used to plot random image in dataset


def showFolderImageSample(base_folder, img_number=2):
    '''
    base_folder(String) : directory in which we'll look for sample
    img_number : for each folder how many img to show
    '''
    nrows = ncols = img_number

    fig = plt.gcf()
    fig.set_size_inches(ncols * img_number, nrows * img_number)
    dir_names = os.listdir(base_folder)

    # get sample directory
    sample_dir = []
    for i in range(img_number):
        sample_dir.append(dir_names[randint(0,len(dir_names)-1)])
    dir_names = [os.path.join(base_folder, dname) for dname in sample_dir]

    print("total sample directory : {}".format(len(dir_names)))
    # for each sample directory get img_number random image

    img = []
    for d in dir_names:
        for i in range(img_number):
            img.append(os.path.join(d, os.listdir(
                d)[randint(0, len(os.listdir(d)) - 1)]))

    for i, img_path in enumerate(img):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)
        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()
    print("Size of one random image : {}".format(img.shape))
    print(" Example image : \n {}".format(img[0]))

########################################################################


def dirTotalFile(base):
    fileNumber = 0
    if(os.path.isfile(base)):
        return 0
    _dir = os.listdir(base)
    for d in _dir:
        _d = os.path.join(base, d)
        if(os.path.isdir(_d)):
            fileNumber += dirTotalFile(_d)
        else:
            fileNumber += 1
    return fileNumber


########################################################################

showFolderImageSample(base_dir,img_number=2)


# train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)
# 
# val_datagen = ImageDataGenerator(rescale=1./255)
# 
# train_generator = train_datagen.flow_from_directory(
#     base_dir,
#     target_size=(128,128),
#     batch_size=30,
#     class_mode="categorical",
#     subset="training"
# )
# 
# validation_generator = train_datagen.flow_from_directory(
#     base_dir,
#     target_size=(128,128),
#     batch_size=30,
#     class_mode="categorical",
#     subset="validation"
# )

# In[19]:


# keras font model
from keras import optimizers

nodes = 32
optimizer = optimizers.rmsprop(lr=0.001)

network = models.Sequential()
network.add(layers.Conv2D(32,3,activation='relu',input_shape=(128,128,3)))
network.add(layers.MaxPooling2D(2))

network.add(layers.Conv2D(64,2,activation="relu", input_shape=(128,128,3)))
network.add(layers.MaxPooling2D(2))

network.add(layers.Flatten())
network.add(layers.Dense(62, activation='softmax'))

network.summary()


# In[20]:



epochs = 10

network.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['acc'])


history = network.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=epochs,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=10
)


history_dict = history.history 
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.rcParams['figure.figsize'] = (16,8) # Make the figures a bit bigger
fig,(ax1,ax2,) = plt.subplots(1,2)

x = range(1,epochs+1)
ax1.plot(x,loss_values,'bo',label='Training Loss')
ax1.plot(x,val_loss_values,'ro',label='Validation Loss')
ax1.set_title('Training and Validation loss ')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()


ax2.plot(x,acc_values,'b',label='Training Accuracy')
ax2.plot(x,val_acc_values,'r',label='Validation Accuracy')
ax2.set_title('Training and Validation accuracy ')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.show()


# In[40]:


test_loss, test_acc = network.evaluate_generator(validation_generator,verbose=1,steps=100)
print('accuracy on test set : {}'.format(test_acc))


# In[ ]:




