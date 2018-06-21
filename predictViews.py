# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 07:57:57 2018

@author: Adrian
"""

import os
import pydicom
import numpy as np
import gzip
from skimage.transform import resize
import imageio


from keras.models import Model, model_from_json, Sequential
from keras.layers import TimeDistributed, Dense, SimpleRNN, Input, Flatten, Convolution2D, InputLayer, Reshape, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam

goalpath=os.path.join(os.getcwd(), 'main_dir')
model_path='View_custom'

if not os.path.isdir('train_data_RWMA'):
    os.makedirs('train_data_RWMA')
save_path=os.path.join(os.getcwd(), 'train_data_RWMA')

ANdirlist=os.listdir(goalpath)


def labeling(x='all', ANdirlist=ANdirlist):
    if x=='all': ANdirlist=ANdirlist[:]
    else: ANdirlist=ANdirlist[:int(x)]
    
    all_labels=[]
    for i, an in enumerate(ANdirlist):
        print('{}/{}'.format(i, len(ANdirlist)))
        savedirpath=os.path.join(save_path, an)
        if not os.path.isdir(savedirpath):
            os.makedirs(savedirpath)
            #print('made dir')
        savedirlist=os.listdir(savedirpath)
        if len(savedirlist)<4:
            labels=ANdir_labeling(an)

        all_labels.append(labels)        
        
    return all_labels

def ANdir_labeling(an):

    an_path=os.path.join(goalpath, an)
    files=os.listdir(an_path)
    
    for i, dcm in enumerate(files):
        print('{} is being processing'.format(i))
        file_path=os.path.join(an_path, dcm)
        try: 
            tar_array=preparefile(file_path)
            #print(tar_array.shape)
            print('Predicting', end='')
            label=model.predict(tar_array)
            print('.....Prediction done')

        except NotImplementedError: 
            label=np.array([[0,0,0,0,0]])
       
        if i==0: labels=label
        else : labels=np.concatenate((labels, label))

    name_list=['LAX', 'SAX', '4ch', '2ch']
    for seq, i in enumerate(labels.argmax(axis=0)[:4]):
        file_path=os.path.join(an_path, files[i])
        tar_array=preparefile(file_path, a=768, b=1024)
        save_file_path=os.path.join(save_path, an, name_list[seq]+'.npy.gz')
        img_file_path=os.path.join(save_path, an, name_list[seq]+'.jpg')
        with gzip.GzipFile(save_file_path, "w") as gf:
            np.save(gf, tar_array)
        imageio.imwrite(img_file_path, tar_array[0][5])

    return labels

def get_array(path):
    ds=pydicom.read_file(path)
    dt=ds.pixel_array
    return dt

def regrid_vid(tar_array, a=600, b=800):
    for i, img in enumerate(tar_array):
        img=regrid(img, a=a, b=b)
        if i==0: tar_array_sub=img
        else : tar_array_sub=np.concatenate((tar_array_sub, img))
    return tar_array_sub

def regrid(img, a=600, b=800):
    img_resized=resize(img, (a,b))
    img_resized=np.expand_dims(img_resized, axis=0)
    return img_resized

def pad2fifty(tar_dcm):
    add_len=50-len(tar_dcm)
    add_array=np.zeros(shape=(add_len, tar_dcm.shape[1], tar_dcm.shape[2], tar_dcm.shape[3]))
    tar_dcm=np.concatenate((tar_dcm, add_array))
    return tar_dcm

def preparefile(file_path, a=600, b=800):
    tar_array=get_array(file_path)/255
    #print(tar_array.shape)
    if len(tar_array.shape) < 4: raise NotImplementedError
    if tar_array.shape[1:4] != (a, b, 3):
        tar_array=regrid_vid(tar_array, a, b)
    if len(tar_array)<50:
        tar_array=pad2fifty(tar_array)   
    tar_array=tar_array[:50]
    tar_array=np.expand_dims(tar_array, axis=0)
    return tar_array

def made_model():
    model=Sequential()
    
    model.add(InputLayer(input_shape=(50, 600, 800, 3)))
    model.add(Reshape((5, 10, 600, 800, 3)))
    model.add(TimeDistributed(TimeDistributed(MaxPooling2D(pool_size=2, strides=2))))

    model.add(TimeDistributed(TimeDistributed(Conv2D(kernel_size=3, strides=1, filters=5, padding='same',
                     activation='relu', name='layer_conv1'))))
    model.add(TimeDistributed(TimeDistributed(MaxPooling2D(pool_size=2, strides=2))))
    model.add(TimeDistributed(TimeDistributed(Dropout(0.5))))
    model.add(TimeDistributed(TimeDistributed(BatchNormalization())))

    model.add(TimeDistributed(TimeDistributed(Conv2D(kernel_size=5, strides=1, filters=20, padding='same',
                     activation='relu', name='layer_conv2'))))
    model.add(TimeDistributed(TimeDistributed(MaxPooling2D(pool_size=2, strides=2))))
    model.add(TimeDistributed(TimeDistributed(Dropout(0.5))))
    model.add(TimeDistributed(TimeDistributed(BatchNormalization())))

    model.add(TimeDistributed(TimeDistributed(Conv2D(kernel_size=5, strides=1, filters=20, padding='same',
                     activation='relu', name='layer_conv2'))))    
    model.add(TimeDistributed(TimeDistributed(MaxPooling2D(pool_size=2, strides=2))))
    model.add(TimeDistributed(TimeDistributed(Dropout(0.5))))

    model.add(TimeDistributed(TimeDistributed(Flatten())))
    model.add(TimeDistributed(TimeDistributed(Dense(256, activation='relu'))))
    
    model.add(TimeDistributed(SimpleRNN(256, return_sequences=False, stateful=False)))
    model.add(SimpleRNN(256))
    model.add(Dense(5, activation='softmax'))
    
    optimizer=Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


if __name__=='__main__':
    x=input('How many folders you wanna predict?\n')
    model=made_model()
    model.load_weights(model_path)
    try:
        x=int(x)
        all_labels=labeling(x=x, ANdirlist=ANdirlist)
    except:
        all_labels=labeling()