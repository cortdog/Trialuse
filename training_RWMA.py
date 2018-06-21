# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 09:31:29 2018

@author: Adrian
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:18:03 2018

@author: Adrian
"""

import tensorflow as tf
import os
import numpy as np
import gzip
from itertools import cycle
from skimage.transform import resize
import pickle

label_path=os.path.join(os.getcwd(), 'Labeling.pickle')
with open(label_path, 'rb') as handle:
    label_dict=pickle.load(handle)
    
    
train_path=os.path.join(os.getcwd(), 'train_data_RWMA')
train_anlist=os.listdir(train_path)

batch_size=5

total_numbers=len(os.listdir(train_path))
train_numbers=1500
val_numbers=total_numbers-train_numbers


iter_train=cycle(range(train_numbers))
iter_val=cycle(range(train_numbers+1, total_numbers))


def regrid(img, x=600, y=800):
    img_resized=resize(img, (x,y))
    return img_resized


def fetch_data(batch_size=batch_size, train=True):
    label_batch=[]
    for i in range(batch_size):
        
        if train: iters=iter_train
        else: iters=iter_val
        

        fol=next(iters)
        an=train_anlist[fol]
        
        anpath=os.path.join(train_path, an)
        
        tar_array=get_array_fromAN(anpath)
        tar_array=np.expand_dims(tar_array, axis=0)
        if i==0:
            train_batch=tar_array
        else:
            train_batch=np.concatenate((train_batch, tar_array))
    
        label_batch.append(label_dict[an])
    
    label_batch=np.array(label_batch)
        
    return train_batch, label_batch
        

def get_array_fromAN(anpath):
    name_list=['LAX', 'SAX', '4ch', '2ch']
    file='{}.npy.gz'
    for i,f in enumerate(name_list):
        view_path=os.path.join(anpath, file.format(f))
        with gzip.GzipFile(view_path, "r") as gf:
            tar_array_sub=np.load(gf)
        tar_array_sub=np.expand_dims(tar_array_sub, axis=0)    
        
        if i==0: tar_array=tar_array_sub
        else : tar_array=np.concatenate((tar_array, tar_array_sub))
    return tar_array


from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, LSTM, SimpleRNN
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import TimeDistributed


def made_model():
    model=Sequential()
    
    model.add(InputLayer(input_shape=(4, 50, 600, 800, 3)))
    model.add(TimeDistributed(TimeDistributed(MaxPooling2D(pool_size=2, strides=2))))
    model.add(TimeDistributed(TimeDistributed(Conv2D(kernel_size=3, strides=1, filters=5, padding='same',
                     activation='relu', name='layer_conv1'))))
    model.add(TimeDistributed(TimeDistributed(MaxPooling2D(pool_size=2, strides=2))))
    model.add(TimeDistributed(TimeDistributed(Conv2D(kernel_size=5, strides=1, filters=20, padding='same',
                     activation='relu', name='layer_conv2'))))
    model.add(TimeDistributed(TimeDistributed(MaxPooling2D(pool_size=2, strides=2))))
    model.add(TimeDistributed(TimeDistributed(Flatten())))
    model.add(TimeDistributed(TimeDistributed(Dense(128, activation='relu'))))
    
    model.add(TimeDistributed(SimpleRNN(64, return_sequences=False, stateful=False)))
    model.add(SimpleRNN(64, return_sequences=False, stateful=False))
    model.add(Dense(6, activation='softmax'))
    
    optimizer=Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def optimize(iter_n):
    for i in range(iter_n):
        x,y=fetch_data()
        model.fit(x,y, epochs=3, batch_size=1)
        
        if i%5==0:
            model.save('model_RWMA')
            print('The {} loop saved'.format(i))


try: model=load_model('model_RWMA')
except:
    print('new model')
    model=made_model()